import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

# App title
st.set_page_config(page_title="💬 Cakap Assistant")
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
