import streamlit as st
from modelo import processar_pergunta

st.set_page_config(page_title="Chatbot Médico com IA", page_icon="💬")
st.title("💬 Chatbot Médico com IA")
st.caption("🚀 Faça perguntas sobre o DataFrame de pacientes")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso ajudar você?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Pensando..."):
        resposta = processar_pergunta(prompt)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.chat_message("assistant").write(resposta)