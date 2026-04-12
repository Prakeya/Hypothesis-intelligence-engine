import streamlit as st
from server.env import HypothesisEnv, Action, Observation, State
import time
import json
from datetime import datetime
import uuid

# --- Page Config ---
st.set_page_config(
    page_title="Hypothesis Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
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
if "mode" not in st.session_state:
    st.session_state.mode = "benchmark"

# --- Custom CSS (Modern Fintech UI) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');

.stApp { 
    background-color: #0E0E10; 
    color: #F0F0F0; 
    font-family: 'Inter', sans-serif; 
}
header, footer, [data-testid="stHeader"] {visibility: hidden;}

/* Ambient Central Core Core */
.block-container::before {
    content: ""; position: fixed; top: 15%; left: 50%; transform: translateX(-50%); 
    width: 350px; height: 60vw; border-radius: 40%;
    background: radial-gradient(ellipse at center, #333333 0%, #555555 30%, #111111 70%, transparent 80%);
    filter: blur(80px); opacity: 0.8; z-index: -10; pointer-events: none;
}


.glass-panel {
    background: #1C1C1E;
    border-radius: 24px;
    padding: 2.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

.hero-title { font-family: 'Outfit', sans-serif; font-size: 5rem; font-weight: 800; color: #FFFFFF; text-align: center; line-height: 1.1; }
.hero-subtitle { font-size: 0.8rem; font-weight: 700; color: #AAAAAA; text-align: center; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 1rem;}

.chapter-tag { font-size: 0.7rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #777; margin-bottom: 1.5rem; }
.chapter-title { font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; margin-bottom: 2rem; }

.stButton > button {
    border-radius: 50px !important; font-weight: 700 !important; font-size: 0.85rem !important; padding: 1.2rem 3rem !important;
    border: 1px solid rgba(255,255,255,0.1) !important; background: transparent !important; color: #A0A0A0 !important;
    backdrop-filter: blur(5px);
}
.stButton > button:hover { background: rgba(255, 255, 255, 0.1) !important; color: white !important; }
.stButton > button[kind="primary"] { 
    background: #FFFFFF !important; 
    color: #0E0E10 !important; 
    border: none !important;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.4);
}

.system-badge {
    background: #28282C;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #AAA;
}

/* Protocol Cards Enhancements */
.protocol-card {
    background: linear-gradient(145deg, #18181A 0%, #121214 100%);
    border: 1px solid rgba(255, 255, 255, 0.04); 
    border-radius: 24px; 
    padding: 3rem 2rem;
    text-align: center; 
    margin-bottom: 2rem; 
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
    min-height: 310px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.protocol-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, #FFFFFF, rgba(255,255,255,0.1));
    opacity: 0.2; transition: opacity 0.3s ease;
}
.protocol-card:hover { transform: translateY(-4px); border-color: rgba(255, 255, 255, 0.2); box-shadow: 0 15px 35px rgba(0,0,0,0.4); }
.protocol-card:hover::before { opacity: 1; }
.protocol-card h3 { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.8rem; margin-bottom: 1rem; color: #FFF; letter-spacing: 0.5px; text-transform: uppercase; }
.protocol-card p { font-family: 'Inter', sans-serif; color: #888; font-size: 0.85rem; line-height: 1.7; margin-bottom: 2rem; font-weight: 300; letter-spacing: 0.3px; }

.mode-tag {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 2.5px; color: #555; margin-bottom: 1.5rem; text-transform: uppercase;
}

.engine-card {
    background: linear-gradient(145deg, #1C1C1E 0%, #17171A 100%);
    border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 2rem;
    height: 100%; margin-bottom: 1rem;
    transition: all 0.2s ease;
}
.engine-card:hover { border-color: rgba(255, 255, 255, 0.15); }
.engine-card h4 { font-family: 'Outfit', sans-serif; font-weight: 600; margin-bottom: 0.5rem; }

/* Logic Node UI */
.logic-node {
    background: #151518;
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.logic-node-header {
    display: flex;
    align-items: baseline;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 0.8rem;
}
.logic-node-step {
    font-size: 0.75rem; color: #666; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; margin-right: 1rem;
}
.logic-node-title {
    font-family: 'Outfit', sans-serif; font-size: 1.3rem; color: #EAEAEA; font-weight: 600;
}
.logic-node-body {
    font-family: 'Inter', sans-serif; color: #BBB; font-size: 1rem; line-height: 1.7; font-weight: 300;
}

/* Annihilate Streamlit's Native React Unmount Lag & Fade Ghosting */
* {
    animation: none !important;
    animation-duration: 0.0s !important; 
    transition: none !important;
    transition-duration: 0.0s !important;
}
[data-testid="stAppViewContainer"] * {
    transition-delay: 0s !important;
    animation-delay: 0s !important;
}
/* Force immediate display removal for blocks that are hiding */
[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}
/* Whitelist our custom premium UI hover states WITHOUT triggering opacity fades */
.protocol-card {
    transition: transform 0.1s ease-out, border-color 0.1s ease-out !important;
}
.protocol-card:hover {
    transform: translateY(-5px) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}
    transition: transform 0.1s ease-out, border-color 0.1s ease-out !important;
}
.glass-panel:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(255,255,255,0.1) !important;
}

/* Semantic Highlighting */
.val-supported { color: #4ade80; font-weight: 700; }
.val-refuted { color: #f87171; font-weight: 700; }
.val-inconclusive { color: #fbbf24; font-weight: 700; }
.trend-inc { color: #60a5fa; font-weight: 600; }
.trend-dec { color: #f472b6; font-weight: 600; }
.trend-mixed { color: #a78bfa; font-weight: 600; }

.semi-bold { font-weight: 600; color: #EEE; }
.premium-bullet { color: #60a5fa; margin-right: 0.8rem; font-size: 1.1rem; }
.bullet-list { list-style-type: none; padding: 0; margin: 0; }
.bullet-item { margin-bottom: 0.8rem; display: flex; align-items: start; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def reset_system():
    st.session_state.entered = False
    st.session_state.mode = None
    st.session_state.agent_output = None
    st.session_state.evaluation = None
    st.session_state.current_obs = st.session_state.env.reset(mode="benchmark")

def init_kernel_cb():
    st.session_state.entered = "active"
    st.session_state.mode = None

def select_bench_cb():
    st.session_state.mode = "benchmark"
    st.session_state.agent_output = None

def select_cust_cb():
    st.session_state.mode = "custom"
    st.session_state.current_obs = st.session_state.env.reset(mode="custom")
    st.session_state.agent_output = None

def go_back_cb():
    st.session_state.mode = None
    st.session_state.agent_output = None

# --- ROUTER ---
@st.dialog("Execution Analysis Output", width="large")
def show_analysis_dialog():
    out = st.session_state.agent_output
    eval_data = st.session_state.evaluation
    obs = st.session_state.current_obs

    st.markdown("<div class='chapter-tag'>Active Claim Under Test</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 1.5rem; font-style: italic; color: #fff; margin-bottom: 2rem;'>\"{obs.claim}\"</div>", unsafe_allow_html=True)

    st.markdown("<div id='logic-anchor' class='chapter-tag'>Audit Conclusion & Status</div>", unsafe_allow_html=True)
    info_data = eval_data.get('info', {})
    reward = eval_data.get('reward', 0.5)

    status_text = "System Initialized"
    status_color = "#FFFFFF"
    
    trend = eval_data.get('info', {}).get('trend', 'neutral')
    
    # 1. Strong cases (Emerald)
    if trend in ["strong_positive", "strong_negative"]:
        status_text = "🏆 STRONG CORRELATION"
        status_color = "#10b981" # emerald-500
        subtext = "A clear directional relationship is consistently visible in the observed data."
    
    # 2. Moderate cases (Sky)
    elif trend in ["positive", "negative"]:
        status_text = "📈 CORRELATION OBSERVED"
        status_color = "#0ea5e9" # sky-500
        subtext = "A stable directional trend is visible in the current evidence."
    
    # 3. Mixed cases (Amber)
    elif trend == "mixed":
        status_text = "⚖️ MIXED EVIDENCE PATTERN"
        status_color = "#f59e0b" # amber-500
        subtext = "The evidence shifts in conflicting directions, limiting a single clean interpretation."
    
    # 4. Weak cases (Zinc)
    else: # neutral
        status_text = "🔍 WEAK SIGNAL"
        status_color = "#71717a" # zinc-500
        subtext = "The current dataset does not provide enough directional strength for a confident pattern claim."

    st.markdown(f"""
    <div class='glass-panel' style='display: flex; justify-content: space-between; align-items: center; border-left: 4px solid {status_color}; background: linear-gradient(90deg, {status_color}0A 0%, rgba(28,28,30,1) 100%);'>
        <div>
            <div style='font-size: 0.7rem; font-weight: 800; color: #888; text-transform: uppercase; letter-spacing: 2px;'>AUDIT VERDICT</div>
            <div style='font-size: 2.8rem; font-family: Outfit, sans-serif; color: #FFF; font-weight: 800; margin-top: 0.2rem;'>{out['verdict']}</div>
        </div>
        <div style='text-align: right;'>
            <div style='font-size: 0.7rem; font-weight: 800; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;'>SYSTEM STATUS</div>
            <div style='display: inline-block; padding: 0.6rem 1.5rem; border-radius: 50px; background: rgba(255,255,255,0.03); border: 1px solid {status_color}44; color: {status_color}; font-family: Outfit, sans-serif; font-weight: 700; font-size: 0.9rem;'>{status_text}</div>
            <div style='font-size: 0.75rem; color: #666; margin-top: 0.5rem; font-weight: 400;'>{subtext}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- EXECUTIVE BREAKDOWN (REFINED) ---
    st.markdown("<div class='chapter-tag' style='margin-top: 2rem;'>Executive Audit Summary</div>", unsafe_allow_html=True)
    
    ind = obs.independent_var
    dep = obs.dependent_var
    v_color = "val-supported" if out['verdict'] == "Supported" else ("val-refuted" if out['verdict'] == "Refuted" else "val-inconclusive")
    
    trend_cls = "trend-inc" if "positive" in trend else ("trend-dec" if "negative" in trend else "trend-mixed")
    if "strong" in trend: trend_label = "Strong " + ("Upward" if "positive" in trend else "Downward")
    elif trend != "mixed" and trend != "neutral": trend_label = "Moderate " + ("Upward" if "positive" in trend else "Downward")
    else: trend_label = trend.capitalize()

    if trend == "strong_positive": trend_desc = "Highly consistent upward correlation"
    elif trend == "strong_negative": trend_desc = "Highly consistent downward correlation"
    elif trend == "positive": trend_desc = "Clear directional trend visible"
    elif trend == "negative": trend_desc = "Inverse relationship visible"
    elif trend == "mixed": trend_desc = "Evidence moves in mixed directions"
    else: trend_desc = "Insufficient directional strength"

    sum_html = f"""
    <div class='glass-panel' style='padding: 2rem;'>
        <ul class='bullet-list'>
            <li class='bullet-item'>
                <span class='premium-bullet'>•</span>
                <span><span style='color:#777;'>Variables Identified:</span> <span class='semi-bold'>{ind}</span> (Cause) & <span class='semi-bold'>{dep}</span> (Effect)</span>
            </li>
            <li class='bullet-item'>
                <span class='premium-bullet'>•</span>
                <span><span style='color:#777;'>Pattern Observed:</span> <span class='{trend_cls}'>{trend_desc}</span></span>
            </li>
            <li class='bullet-item'>
                <span class='premium-bullet'>•</span>
                <span><span style='color:#777;'>Logic Check:</span> <span class='{v_color}'>{"Claim aligns with evidence" if out['verdict'] == "Supported" else ("Claim contradicts evidence" if out['verdict'] == "Refuted" else "Claim lacks empirical grounding")}</span></span>
            </li>
        </ul>
    </div>
    """
    st.markdown(sum_html, unsafe_allow_html=True)

    st.markdown("<div class='chapter-tag' style='margin-top: 2rem;'>Formal Logic Trace</div>", unsafe_allow_html=True)
    
    import re
    reasoning_text = out['reasoning']
    steps = reasoning_text.split('Step ')
    for step in steps:
        if not step.strip(): continue
        parts = step.split('\n', 1)
        title_parts = parts[0].split(':', 1)
        
        # Premium Step Title Aliases
        title_map = {
            "STEP 1": "Metric Isolation",
            "STEP 2": "Evidence Validation",
            "STEP 3": "Empirical Trend Analysis",
            "STEP 4": "Hypothesis Direction",
            "STEP 5": "Synthesized Conclusion"
        }
        
        raw_step_tag = f"STEP {title_parts[0].strip()}"
        step_alias = title_map.get(raw_step_tag.upper(), raw_step_tag)
        title_text = title_parts[1].strip() if len(title_parts) > 1 else title_parts[0].strip()
        
        body_raw = parts[1] if len(parts) > 1 else ""
        
        # Convert Markdown Bold to HTML semi-bold
        body_html = re.sub(r'\*\*(.*?)\*\*', r'<span class="semi-bold">\1</span>', body_raw)
        
        # Structure sentences as bullets
        sentences = [s.strip() for s in body_html.split('. ') if s.strip()]
        bullet_html = "<ul class='bullet-list'>"
        for s in sentences:
            if not s.endswith('.'): s += '.'
            bullet_html += f"<li class='bullet-item'><span class='premium-bullet' style='font-size:0.7rem; margin-top:0.4rem;'>•</span><span>{s}</span></li>"
        bullet_html += "</ul>"
        
        html_block = f"""
        <div class='logic-node'>
            <div class='logic-node-header'>
                <span class='logic-node-step'>{raw_step_tag.upper()}</span>
                <span class='logic-node-title' style='text-transform: uppercase; letter-spacing: 1px; font-size: 1rem; color: #FFF;'>— {step_alias.upper()}</span>
            </div>
            <div class='logic-node-body'>{bullet_html}</div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)

    st.markdown("<div class='chapter-tag' style='margin-top: 3rem;'>Final AI Synthesis</div>", unsafe_allow_html=True)
    
    if out['verdict'] == "Supported":
        c_text = f"The evidence demonstrates a dominant and verifiable correlation. The analytical kernel has confirmed that the dataset behavior aligns exactly with the hypothesis parameters. The claim is decisively **Supported**."
    elif out['verdict'] == "Refuted":
        c_text = f"The empirical data demonstrates a clear contradiction to the hypothesis. The observed trend moves in opposition to the expected outcome, suggesting the claim is fundamentally **Refuted** by the current dataset."
    else:
        c_text = f"The dataset exhibits significant volatility or a lack of directional clarity. No stable correlation can be isolated to support or refute the hypothesis, rendering the claim **Inconclusive** within this evidentiary scope."

    st.markdown(f"<div class='glass-panel' style='border-left: 5px solid #60a5fa; background: linear-gradient(90deg, rgba(96, 165, 250, 0.05) 0%, rgba(28,28,30,1) 100%);'><p style='font-size: 1.15rem; font-weight: 400; line-height: 1.7; color:#EAEAEA; margin: 0; font-family: Inter, sans-serif;'>{c_text}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='chapter-tag' style='margin-top: 4rem; text-align: center;'>Reward Allocation</div>", unsafe_allow_html=True)
    
    info_data = eval_data.get('info', {})
    breakdown_html = ""
    if 'breakdown' in info_data:
        breakdown_html += "<div style='display: flex; flex-direction: column; max-width: 600px; margin: 0 auto 2rem auto; background: rgba(255,255,255,0.02); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.05);'>"
        for b in info_data['breakdown']:
            status_color = "#4ade80" if "PASS" in b['status'] else ("#f87171" if "FAIL" in b['status'] else "#777")
            bg_color = "rgba(74, 222, 128, 0.1)" if "PASS" in b['status'] else ("rgba(248, 113, 113, 0.1)" if "FAIL" in b['status'] else "rgba(255,255,255,0.05)")
            breakdown_html += f"<div style='margin-bottom:0.8rem; padding-bottom:0.8rem; border-bottom:1px solid rgba(255,255,255,0.02);'>"
            breakdown_html += f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;'>"
            breakdown_html += f"<span style='color:#EAEAEA; font-size:1rem; font-weight:600; font-family: Inter, sans-serif;'>{b['metric']}</span>"
            breakdown_html += f"<span style='color:{status_color}; background: {bg_color}; padding: 4px 12px; border-radius: 20px; font-family:Outfit,sans-serif; font-weight:700; font-size:0.9rem;'>{b['points']} <span style='opacity:0.7; font-size:0.75rem;'>{b['status']}</span></span>"
            breakdown_html += "</div>"
            if 'reason' in b:
                breakdown_html += f"<div style='color:#888; font-size:0.8rem; font-family: Inter, sans-serif; line-height: 1.4;'>{b['reason']}</div>"
            breakdown_html += "</div>"
        breakdown_html += "</div>"

    st.markdown(breakdown_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    # Qualitative Labels for Scorecard
    if reward >= 0.85: g_label = "Optimal"
    elif reward >= 0.6: g_label = "Accurate"
    else: g_label = "Suboptimal"
    
    conf = out.get('confidence', 0.5)
    if conf >= 0.85: depth_label = "Exhaustive"
    elif conf >= 0.6: depth_label = "Balanced"
    else: depth_label = "Cursory"

    with col1:
         st.markdown(f"<div class='glass-panel' style='text-align: center; padding: 1.5rem;'><div class='chapter-tag' style='margin-bottom:0.5rem; font-size: 0.6rem;'>LOGICAL GROUNDING</div><div style='font-size:2.5rem; font-family:Outfit,sans-serif; font-weight:800; color:#FFF; line-height:1;'>{reward:.2f}</div><div style='font-size:0.7rem; color:#888; margin-top:0.5rem; font-weight:700;'>{g_label.upper()}</div></div>", unsafe_allow_html=True)
    with col2:
         st.markdown(f"<div class='glass-panel' style='text-align: center; padding: 1.5rem;'><div class='chapter-tag' style='margin-bottom:0.5rem; font-size: 0.6rem;'>INFERENCE DEPTH</div><div style='font-size:2.5rem; font-family:Outfit,sans-serif; font-weight:800; color:#FFF; line-height:1;'>{conf:.2f}</div><div style='font-size:0.7rem; color:#888; margin-top:0.5rem; font-weight:700;'>{depth_label.upper()}</div></div>", unsafe_allow_html=True)
    with col3:
         # Use the dynamic status color for the badge here too
         st.markdown(f"<div class='glass-panel' style='text-align: center; padding: 1.5rem;'><div class='chapter-tag' style='margin-bottom:0.5rem; font-size: 0.6rem;'>AUDIT FIDELITY</div><div style='font-size:1.2rem; font-family:Outfit,sans-serif; font-weight:700; color:{status_color}; height:2.5rem; display:flex; align-items:center; justify-content:center; line-height:1.2;'>{status_text}</div><div style='font-size:0.7rem; color:#888; margin-top:0.5rem; font-weight:700;'>VERIFIED</div></div>", unsafe_allow_html=True)

# --- VIEW CONTAINERS TO PREVENT GHOSTING ---
home_placeholder = st.empty()
protocol_placeholder = st.empty()

if not st.session_state.entered:
    with home_placeholder.container():
        st.markdown("<div style='height: 25vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>OpenEnv Logic Auditing Engine</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>Hypothesis Intelligence</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 12rem;'></div>", unsafe_allow_html=True)
        _, col_btn, _ = st.columns([1, 0.6, 1])
        with col_btn:
            st.button("Initialize Kernel", type="primary", use_container_width=True, on_click=init_kernel_cb)

else:
    # INSTANT DESTRUCTION of home elements
    home_placeholder.empty()
    
    if st.session_state.mode:
        # If in a mode, immediately destroy the protocol selection UI
        protocol_placeholder.empty()
        st.button("← Change Protocol", on_click=go_back_cb)
    else:
        st.button("← Power Off", on_click=reset_system)

    # 1. Protocol Selection
    if not st.session_state.mode:
        with protocol_placeholder.container():
            st.markdown("<div class='chapter-title' style='text-align:center; margin-top:2rem;'>Protocol Selection</div>", unsafe_allow_html=True)
            st.markdown("<div class='chapter-subtitle' style='text-align:center; color:#AAA; margin-bottom: 2rem;'>Select an operating configuration below to calibrate the engine limits.</div>", unsafe_allow_html=True)
            
            col_bench, col_cust = st.columns(2)
            with col_bench:
                bg = "linear-gradient(145deg, #18181A 0%, #121214 100%)"
                accent = "<div style='width: 35px; height: 2px; background: rgba(255,255,255,0.2); margin: 0 auto 1.5rem auto;'></div>"
                
                st.markdown(f"""
                <div class='protocol-card' style='border-color: rgba(255,255,255,0.04); background: {bg};'>
                    <div class='mode-tag'>MODE [01]</div>
                    {accent}
                    <h3>Benchmark Runtime</h3>
                    <p>Run pre-calibrated evaluation tasks and inspect structured reasoning under benchmark conditions.</p>
                </div>
                """, unsafe_allow_html=True)
                st.button("Initialize Benchmark Stack", use_container_width=True, type="secondary", on_click=select_bench_cb)
                    
            with col_cust:
                bg = "linear-gradient(145deg, #18181A 0%, #121214 100%)"
                accent = "<div style='width: 35px; height: 2px; background: rgba(255,255,255,0.1); margin: 0 auto 1.5rem auto;'></div>"
                
                st.markdown(f"""
                <div class='protocol-card' style='border-color: rgba(255,255,255,0.04); background: {bg};'>
                    <div class='mode-tag'>MODE [02]</div>
                    {accent}
                    <h3>Custom Injector</h3>
                    <p>Inject custom datasets into the inference pipeline and inspect open-ended analytical behavior.</p>
                </div>
                """, unsafe_allow_html=True)
                st.button("Initialize Custom Injector", use_container_width=True, type="secondary", on_click=select_cust_cb)

    # 2. Engine Selection & Preview
    elif st.session_state.mode == "benchmark":
        st.markdown("<div class='chapter-title' style='text-align:center;'>Target Strategy Engine</div>", unsafe_allow_html=True)
        
        all_tasks = st.session_state.env.benchmark_tasks
        cols = st.columns(3)
        for i, t in enumerate(all_tasks):
            with cols[i % 3]:
                domain = t.get("domain", "General")
                badge_html = f"<span style='background:rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; color: #FFF; font-weight: 600; text-transform: uppercase;'>{domain}</span>"
                st.markdown(f"<div class='engine-card' style='display: flex; flex-direction: column; justify-content: space-between;'><div style='margin-bottom: 1rem;'>{badge_html}</div><div style='color:#EAEAEA; font-size:1.05rem; line-height:1.5; font-weight: 500; margin-bottom: 1rem;'>{t['claim']}</div></div>", unsafe_allow_html=True)
                if st.button("EXECUTE AUDIT", key=f"btn_{t['id']}", use_container_width=True):
                    # Load the environment and auto-execute the agent
                    st.session_state.current_obs = Observation(
                        mode_identifier="benchmark", task_id=t["id"], claim=t["claim"],
                        evidence_block=t["dataset"], independent_var=t["independent_var"], dependent_var=t["dependent_var"]
                    )
                    st.session_state.env._current_state = State(current_task=st.session_state.current_obs)
                    
                    with st.spinner("Executing Logic Audit..."):
                        from server.agent import HypothesisAgent
                        agent = HypothesisAgent(use_llm=False)
                        action_dict = agent.generate_action(st.session_state.current_obs.dict(), st.session_state.audit_id)
                        action = Action(**action_dict)
                        reward = st.session_state.env.step(action)
                        
                        st.session_state.agent_output = action_dict
                        st.session_state.evaluation = reward.dict()
                    show_analysis_dialog()

    elif st.session_state.mode == "custom":
        st.markdown("<div class='chapter-title' style='text-align:center;'>Custom Injector</div>", unsafe_allow_html=True)
        obs = st.session_state.current_obs
        with st.container():
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.9rem; color:#888; margin-bottom:1rem; padding:1rem; border-left:3px solid #AAA; background:rgba(255,255,255,0.02);'><strong>Instructions:</strong> Provide the evidence backing the hypothesis as a valid JSON array of objects. Each object represents an observation, containing keys corresponding to the X-Axis and Y-Axis variables. Only numerical associations undergo strict monotonicity math processing. Example: <code>[{\"hours\": 10, \"marks\": 50}, {\"hours\": 15, \"marks\": 70}]</code></div>", unsafe_allow_html=True)
            custom_claim = st.text_area("Hypothesis Claim", value=obs.claim, height=80)
            custom_dataset_json = st.text_area("Evidence Matrix (JSON Format)", value=json.dumps(obs.evidence_block, indent=2), height=250)
            col_v1, col_v2 = st.columns(2)
            with col_v1: custom_indep = st.text_input("X-Axis Variable", value=obs.independent_var)
            with col_v2: custom_dep = st.text_input("Y-Axis Variable", value=obs.dependent_var)
            
            if st.button("EXECUTE AUDIT ON RAW DATA", type="primary", use_container_width=True):
                try: final_dataset = json.loads(custom_dataset_json)
                except: final_dataset = obs.evidence_block
                
                st.session_state.current_obs.claim = custom_claim
                st.session_state.current_obs.evidence_block = final_dataset
                st.session_state.current_obs.independent_var = custom_indep
                st.session_state.current_obs.dependent_var = custom_dep
                st.session_state.env._current_state = State(current_task=st.session_state.current_obs)
                
                with st.spinner("Executing Logic Audit..."):
                    from server.agent import HypothesisAgent
                    agent = HypothesisAgent(use_llm=False)
                    action_dict = agent.generate_action(st.session_state.current_obs.dict(), st.session_state.audit_id)
                    action = Action(**action_dict)
                    reward = st.session_state.env.step(action)
                    st.session_state.agent_output = action_dict
                    st.session_state.evaluation = reward.dict()
                show_analysis_dialog()

    # 3. Final Conclusion Display Button
    if st.session_state.agent_output:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 3rem 0; '>", unsafe_allow_html=True)
        if st.button("View Previous Analysis Results", use_container_width=True, type="secondary"):
            show_analysis_dialog()

def main():
    import subprocess
    import sys
    from pathlib import Path
    app_path = Path(__file__).absolute()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    main()

