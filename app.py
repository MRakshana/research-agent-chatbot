# app.py

import streamlit as st
import research_agent as ra

st.set_page_config(page_title="Research Agent Chatbot", layout="wide")
st.title("Research Agent Chatbot")
st.caption(
    "Answers are based only on JPMorgan, Pfizer, and Google research documents plus live market data for these companies."
)

# load index once
if "index" not in st.session_state:
    st.session_state.index = ra.build_index()

index = st.session_state.index

# chat history
if "history" not in st.session_state:
    st.session_state.history = []

# input box
question = st.chat_input("Ask about JPMorgan, Pfizer, or Google")

if question:
    user_q = question.strip()
    lower_q = user_q.lower()

    detail = False
    base_question = user_q

    # follow up phrases that mean explain more
    followup_phrases = {
        "explain more",
        "tell me more",
        "more detail",
        "more details",
        "explain in detail",
    }

    if st.session_state.history and lower_q in followup_phrases:
        base_question = st.session_state.history[-1]["user"]
        detail = True

    answer = ra.smart_answer(index, base_question, detail=detail)
    st.session_state.history.append({"user": user_q, "bot": answer})

# display conversation
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        st.write(msg["bot"])
