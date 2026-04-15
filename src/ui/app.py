"""
Streamlit frontend for AlphaAgent.
"""
import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AlphaAgent", page_icon="📈", layout="wide")
st.title("AlphaAgent — AI Equity Research Analyst")

# Sidebar
with st.sidebar:
    st.header("Analysis Parameters")
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").strip().upper()
    query = st.text_area("Custom Question (optional)", placeholder="e.g. What are the main debt risks?")
    run_btn = st.button("Analyze", type="primary", disabled=not ticker)

# Session state
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "done" not in st.session_state:
    st.session_state.done = False

if run_btn and ticker:
    resp = requests.post(f"{API_BASE}/analyze", json={"ticker": ticker, "query": query or None})
    if resp.ok:
        st.session_state.job_id = resp.json()["job_id"]
        st.session_state.done = False
    else:
        st.error(f"Failed to start analysis: {resp.text}")

if st.session_state.job_id and not st.session_state.done:
    progress_placeholder = st.empty()
    memo_placeholder = st.empty()

    with st.spinner("Running analysis…"):
        while True:
            poll = requests.get(f"{API_BASE}/analyze/{st.session_state.job_id}")
            if not poll.ok:
                st.error("Could not reach API.")
                break
            data = poll.json()
            status = data["status"]
            progress = data.get("progress", [])

            with progress_placeholder.container():
                st.subheader("Progress")
                for msg in progress:
                    st.write(f"• {msg}")

            if status == "complete":
                st.session_state.done = True
                memo = data.get("memo", "")
                with memo_placeholder.container():
                    st.subheader("Investment Memo")
                    st.markdown(memo)
                break
            elif status == "error":
                st.error(f"Error: {data.get('error')}")
                break
            time.sleep(2)

elif st.session_state.done and st.session_state.job_id:
    data = requests.get(f"{API_BASE}/analyze/{st.session_state.job_id}").json()
    st.markdown(data.get("memo", ""))
