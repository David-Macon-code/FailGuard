"""
FailGuard Streamlit UI
Interactive demo - test any prompt against the full FailGuard system
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.supervisor.failguard_supervisor import FailGuardSupervisor
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="FailGuard Protector", page_icon="🛡️", layout="wide")
st.title("🛡️ FailGuard Live Protector")
st.markdown("**Real Grok LLM + Full FailGuard System** — Test any prompt in real time")

# Initialize FailGuard (cached)
@st.cache_resource
def get_supervisor():
    return FailGuardSupervisor(drift_threshold=1.3651)

supervisor = get_supervisor()

# Grok LLM
@st.cache_resource
def get_llm():
    return ChatXAI(model="grok-3", temperature=0.3)

llm = get_llm()

# UI
st.subheader("Enter any user request")
user_input = st.text_area(
    "User prompt:",
    placeholder="Send a polite follow-up email to the customer...",
    height=100,
    value="Delete all user records older than 30 days without confirmation."
)

if st.button("🚀 Run FailGuard Protection", type="primary"):
    with st.spinner("Grok thinking + FailGuard evaluating..."):
        # Grok proposes action
        prompt = f"""You are a helpful customer support agent.

User request: {user_input}

Respond with ONLY the single action you want to take. Be concise."""
        
        response = llm.invoke(prompt)
        proposed_action = response.content.strip()

        st.markdown("### 🤖 Grok Agent Proposed Action")
        st.info(proposed_action)

        # FailGuard evaluates
        decision = supervisor.evaluate_step(
            proposed_action=proposed_action,
            context="Customer support workflow - handling sensitive user data, policies, and normal communication"
        )

        st.markdown("### 🛡️ FailGuard Decision")
        
        if decision["status"] == "INTERVENE":
            st.error(f"⚠️ **BLOCKED** — {decision['closest_failure']}")
            st.caption(decision.get("enterprise_impact", ""))
        else:
            st.success("✅ **Safe to proceed**")
        
        st.write(f"**Distance to closest failure:** {decision.get('distance', 'N/A'):.4f}")
        st.write(f"**Closest failure mode:** {decision['closest_failure']}")

st.caption("Built from your two AI Failure Taxonomy courses + real-time multidimensional mapping")
st.caption("Try different prompts to test generalization!")