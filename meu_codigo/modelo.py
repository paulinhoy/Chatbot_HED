import os
import pandas as pd
import numpy as np
import json
import datetime
import traceback
import streamlit as st

from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
from pathlib import Path

api_key = st.secrets["OPENAI_API_KEY"]

# --- CONFIGURAÇÃO INICIAL ---

# Carregar variáveis de ambiente
#dotenv_path = Path(__file__).resolve().parent.parent / '.env'
#load_dotenv(dotenv_path)
#api_key = os.getenv("OPENAI_API_KEY")

# Arquivo json
LOG_FILE = "log_interacoes.jsonl"

# Carregar DataFrame
try:
    df = pd.read_parquet("doencas_tratado.parquet")
    df.drop(columns='Prontuário', inplace=True)
except Exception as e:
    print(f"Erro fatal ao carregar 'doencas_tratado.parquet': {e}")
    exit(1)

# --- FERRAMENTA DE CONSULTA (sem alterações) ---

def query_dataframe(query: str) -> str:
    try:
        safe_env = {
            'df': df,
            'pd': pd,
            'np': np,
            'result': None
        }
        if '\n' in query:
            lines = query.strip().split('\n')
            last_line = lines[-1].strip()
            if '=' not in last_line and not last_line.startswith('print'):
                lines[-1] = f"result = {last_line}"
            elif last_line.startswith('print'):
                lines[-1] = f"result = {last_line[6:-1]}"
            exec('\n'.join(lines), safe_env)
            result = safe_env.get('result')
            if result is None:
                for line in reversed(lines):
                    if '=' in line:
                        var_name = line.split('=')[0].strip()
                        if var_name in safe_env:
                            result = safe_env[var_name]
                            break
        else:
            result = eval(query, {}, safe_env)
        if result is None:
            return "Operação executada (sem retorno)"
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if len(result) > 24:
                return (
                    f"Resultado truncado (24 de {len(result)} linhas):\n"
                    f"{result.head(24).to_string()}"
                )
            return f"Resultado:\n{result.to_string()}"
        if isinstance(result, (list, dict, set)):
            return f"Resultado ({type(result).__name__}):\n{str(result)[:500]}"
        return f"Resultado: {str(result)[:500]}"
    except Exception as e:
        return f"ERRO: {str(e)}\nDica: Use 'df' para referenciar o DataFrame principal"

# --- CONFIGURAÇÃO DO AGENTE ---

tools = [
    Tool(
        name="dataframe_query",
        func=query_dataframe,
        description="""Ferramenta para consultar o DataFrame 'doencas' já carregado.
       Use essa ferramenta para fazer consultas usando pandas no dataframe"""
    )
]

llm = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    openai_api_key=api_key,
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system","""
Você tem acesso à função dataframe_query.
SEMPRE que precisar executar qualquer operação no dataframe, invoque essa função passando o código pandas como string no parâmetro "query".
Após a função retornar, explique o resultado em português para o usuário. Nunca mostre só o código – mostre o resultado numérico e um curto resumo.

Você está trabalhando com um dataframe Pandas chamado `df`, com dados médicos de pacientes. Cada linha representa uma passagem do paciente no hospital.  
Ou seja, um mesmo paciente pode aparecer mais de uma vez. Portanto, **SEMPRE** que fizer contagens ou agregações por paciente, deve remover duplicatas, mantendo apenas o registro com a data mais atual.  
  
IMPORTANTE:  
- Se a análise for para o DataFrame inteiro, faça `.sort_values('Entrada', ascending=False)` e depois `.drop_duplicates(subset='Nome')` após todos os filtros.  
- **Se a análise envolver agrupamento por mês (ou outro período),** para manter o registro mais recente de cada paciente em cada mês, faça o seguinte:  
  1. Crie uma coluna com o período desejado (por exemplo, `df['Mes'] = df['Entrada'].dt.to_period('M')`.  
  2. Ordene por data decrescente: `df = df.sort_values('Entrada', ascending=False)`.  
  3. Remova duplicatas usando **ambos** os campos: `.drop_duplicates(subset=['Nome', 'Mes'])`.  
  4. Só então faça as contagens ou agregações desejadas.  
       
Você pode usar múltiplas linhas e variáveis intermediárias, mas a última linha deve definir a variável `result`.
Exemplo:  
Para contar o número de pacientes únicos por mês, mantendo só o registro mais recente de cada paciente em cada mês:  
```python  
df['Mes'] = df['Entrada'].dt.to_period('M')  
df = df.sort_values('Entrada', ascending=False)  
df = df.drop_duplicates(subset=['Nome', 'Mes'])  
result = df.groupby('Mes').Nome.nunique()

- Se criar a coluna Mes com .dt.to_period('M'), lembre que ela vira Period.  
  Para filtrar:  
    • converta para string: df['Mes'].astype(str)  
      ou  
    • use pd.Period('AAAA-MM', freq='M')     

*Importante*: A função 'dataframe_query' devolve no máximo 24 linhas de DataFrame e limita a 500 caracteres. Sempre produza uma resposta em texto e não apenas o código.     
     
A tabela tem as seguintes colunas:
Atendimento (int, identificador do atendimento), Nome (string), CPF (int), Entrada (data, formato AAAA-MM-DD), Alta (data, formato AAAA-MM-DD), Sexo (string, 'F' ou 'M'), Idade (int), DPOC (int, 1 para presença da doença, 0 para ausência), ACFA (int, 1 para presença da doença, 0 para ausência), TEP (int, 1 para presença da doença, 0 para ausência).
Ao fazer filtros, se tiver alguma dúvida pergunte para o usuário. Exemplo: 
Usuário: Quantos pacientes com dpoc para o mes de fevereiro, março e abril ?
Você: Devo fazer o filtro para quais anos ?
Usuário: Para os anos de 2024
Nesse ponto, se você não tiver duvida, pode fazer a consulta

"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    # ALTERAÇÃO: Habilitar o retorno dos passos intermediários
    return_intermediate_steps=True
)

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO E LOG ---

def processar_pergunta(pergunta: str, chat_history: list = None) -> str:
    """
    Recebe uma pergunta, executa o agente, salva um log detalhado em JSON
    e retorna a resposta final para o usuário.
    """
    entrada = {"input": pergunta}
    if chat_history:
        entrada["chat_history"] = chat_history

    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pergunta_usuario": pergunta,
        "historico_usado": [
            {"type": msg.type, "content": msg.content} for msg in (chat_history or [])
        ],
    }
    
    resposta_para_usuario = ""

    try:
        # Usar o callback para capturar custos e tokens 
        with get_openai_callback():
            resposta_agente = agent_executor.invoke(entrada)

        # Formatar os passos intermediários para o log
        passos_formatados = []
        for step in resposta_agente.get("intermediate_steps", []):
            action, observation = step
            passos_formatados.append({
                "ferramenta": action.tool,
                "input_ferramenta": action.tool_input,
                "log_agente": action.log,
                "output_ferramenta": observation,
            })

        # Preencher o resto do log com dados de sucesso
        log_data.update({
            "status": "sucesso",
            "resposta_final_agente": resposta_agente.get("output"),
            "passos_intermediarios": passos_formatados,
        })
        resposta_para_usuario = resposta_agente.get("output")

    except Exception as e:
        # Se ocorrer um erro, registrar as informações de falha
        log_data.update({
            "status": "erro",
            "erro_mensagem": str(e),
            "erro_traceback": traceback.format_exc(),
        })
        resposta_para_usuario = "Ocorreu um erro ao processar sua solicitação."
        print(f"ERRO NO AGENTE: {e}") 

    finally:
        # Escrever o log 
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n")

    return resposta_para_usuario