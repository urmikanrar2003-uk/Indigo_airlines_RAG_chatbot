import streamlit as st
from main import get_response

st.set_page_config(page_title="Indigo Airline Assistant", layout="centered")

st.title("✈️ Indigo Airline Chat Assistant")
st.markdown("Ask any question about IndiGo.")

# Sidebar

with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat messages

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)

# 👇 This ensures space so input shows properly

st.markdown("---")

# Input (ALWAYS at bottom)

question = st.chat_input("Ask your question here...")

if question and question.strip():
    st.session_state.messages.append(("user", question))
    
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        try:
            answer = get_response(question)
        except Exception as e:
            answer = f"Error: {str(e)}"

    st.session_state.messages.append(("assistant", answer))

    with st.chat_message("assistant"):
        st.write(answer)
