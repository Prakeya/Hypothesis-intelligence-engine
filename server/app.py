import streamlit as st
from server.env import HypothesisEnv, Action, Observation
import time
import json
from datetime import datetime
import uuid

# --- Page Config ---
st.set_page_config(
    page_title="Hypothesis Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- State Management ---
if "entered" not in st.session_state:
    st.session_state.entered = False
if "env" not in st.session_state:
    st.session_state.env = HypothesisEnv()
if "current_obs" not in st.session_state:
    st.session_state.current_obs = None
if "agent_output" not in st.session_state:
    st.session_state.agent_output = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "audit_id" not in st.session_state:
    st.session_state.audit_id = str(uuid.uuid4())[:8].upper()

# --- Custom CSS (Glassy Professional) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Inter:wght@100;200;300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

.stApp { background-color: #08090A; color: #E2E2E2; font-family: 'Inter', sans-serif; }
header, footer, [data-testid="stHeader"] {visibility: hidden;}

.glass-panel {
    background: rgba(255, 255, 255, 0.01);
    backdrop-filter: blur(15px);
    border-radius: 12px;
    padding: 3rem;
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-bottom: 2rem;
}

.hero-title { font-family: 'Lora', serif; font-size: 5rem; font-weight: 700; color: #FFFFFF; text-align: center; }
.hero-subtitle { font-size: 0.7rem; font-weight: 900; color: #666; text-align: center; letter-spacing: 5px; text-transform: uppercase; }

.chapter-tag { font-size: 0.6rem; font-weight: 900; letter-spacing: 6px; text-transform: uppercase; color: #444; margin-bottom: 1.5rem; }
.chapter-title { font-family: 'Lora', serif; font-size: 3.5rem; color: #FFFFFF; line-height: 1.1; margin-bottom: 2rem; }

.stButton > button {
    border-radius: 4px !important; font-weight: 800 !important; font-size: 0.65rem !important; padding: 1rem 3rem !important;
    border: 1px solid rgba(255,255,255,0.1) !important; background: transparent !important; color: #A0A0A0 !important;
}
.stButton > button:hover { background: rgba(255, 255, 255, 0.05) !important; color: white !important; }
.stButton > button[kind="primary"] { background: #FFFFFF !important; color: #08090A !important; }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def reset_system():
    st.session_state.entered = False
    st.session_state.agent_output = None
    st.session_state.evaluation = None
    st.session_state.current_obs = st.session_state.env.reset()
    st.rerun()

# --- ROUTER ---
if not st.session_state.entered:
    st.markdown("<div style='height: 25vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>OpenEnv Logic Auditing Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Hypothesis Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 12rem;'></div>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 0.6, 1])
    with col_btn:
        if st.button("Initialize Kernel", type="primary", use_container_width=True):
            st.session_state.entered = "active"
            st.session_state.current_obs = st.session_state.env.reset()
            st.rerun()

else:
    obs = st.session_state.current_obs

    # --- SIDEBAR (Logic Calibration) ---
    with st.sidebar:
        st.markdown("<div class='chapter-tag'>Logic Calibration</div>", unsafe_allow_html=True)
        use_custom = st.toggle("Manual Override", value=False, help="Invert the kernel logic to use custom artifacts.")
        
        if use_custom:
            st.markdown("---")
            custom_claim = st.text_area("Hypothesis Claim", value=obs.claim, height=100)
            st.markdown("<div style='font-size: 0.7rem; color: #666; margin-bottom: 0.5rem;'>FACT MATRIX (JSON)</div>", unsafe_allow_html=True)
            custom_dataset_json = st.text_area("truth_matrix_entry", value=json.dumps(obs.dataset, indent=2), height=250, label_visibility="collapsed")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                custom_indep = st.text_input("X-Axis", value=obs.independent_var)
            with col_v2:
                custom_dep = st.text_input("Y-Axis", value=obs.dependent_var)
            
            try:
                final_dataset = json.loads(custom_dataset_json)
                st.success("Matrix Validated", icon="✅")
            except Exception as e:
                st.error("Matrix Format Error", icon="❌")
                final_dataset = obs.dataset
        else:
            custom_claim = obs.claim
            final_dataset = obs.dataset
            custom_indep = obs.independent_var
            custom_dep = obs.dependent_var

    col_nav, col_stat = st.columns([1.5, 7.5])
    with col_nav:
        if st.button("← RESET"): reset_system()
    with col_stat:
        st.markdown(f"<div style='text-align:right;'><div class='system-badge'>ID: {st.session_state.audit_id}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 8vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='chapter-tag'>Subject Inquiry</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chapter-title'>\"{custom_claim}\"</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chapter-tag'>Dataset Artifacts</div>", unsafe_allow_html=True)
    st.table(final_dataset)
    
    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)
    if st.button("EXECUTE AUDIT", type="primary"):
        with st.spinner("Decoding reasoning artifacts..."):
            st.session_state.current_obs.claim = custom_claim
            st.session_state.current_obs.dataset = final_dataset
            st.session_state.current_obs.independent_var = custom_indep
            st.session_state.current_obs.dependent_var = custom_dep
            if hasattr(st.session_state.env, '_current_state') and st.session_state.env._current_state:
                st.session_state.env._current_state.current_task = st.session_state.current_obs
                
            time.sleep(1)
            # Baseline simulation call
            from server.agent import HypothesisAgent
            agent = HypothesisAgent(use_llm=True)
            # Use dictionary representation for passing to agent
            task_dict = {
                "task_id": st.session_state.current_obs.task_id,
                "claim": custom_claim,
                "dataset": final_dataset,
                "independent_var": custom_indep,
                "dependent_var": custom_dep
            }
            action_dict = agent.generate_action(task_dict, st.session_state.audit_id)
            
            # Map to OpenEnv Action
            action = Action(**action_dict)
            reward = st.session_state.env.step(action)
            
            st.session_state.agent_output = action_dict
            st.session_state.evaluation = reward.dict()
            st.rerun()

    if st.session_state.agent_output:
        out = st.session_state.agent_output
        eval_data = st.session_state.evaluation
        
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='chapter-tag'>Final Synthesis Verdict</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chapter-title'>{out['conclusion']}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='chapter-tag'>Reasoning Logic</div>", unsafe_allow_html=True)
        st.code(out['reasoning_steps'], language="text")
        
        col1, col2 = st.columns(2)
        with col1:
             st.markdown(f"<div class='glass-panel'>REWARD: {eval_data['reward']}</div>", unsafe_allow_html=True)
        with col2:
             st.markdown(f"<div class='glass-panel'>HALLUCINATION: {eval_data['info']['hallucination_detected']}</div>", unsafe_allow_html=True)

def main():
    import subprocess
    import sys
    from pathlib import Path
    app_path = Path(__file__).absolute()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    main()
