import streamlit as st
from Escondido import E0

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiBorder · Customs Optimizer",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050d1a !important;
    color: #c8d8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 50%, #0a1628 0%, #050d1a 60%) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 0 !important; max-width: 1200px; }

/* ── HEADER ── */
.header-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 0 0 24px 24px;
    margin-bottom: 2.5rem;
    padding: 2.8rem 3rem 2rem;
    background: linear-gradient(135deg, #7b1a2a 0%, #9e2a3a 25%, #1a6b7a 65%, #0d9eaf 100%);
    box-shadow: 0 8px 48px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.12);
}

/* rotating lighter blob */
.header-wrap::before {
    content: '';
    position: absolute;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
    top: -80px; right: -60px;
    animation: spinBlob 8s linear infinite;
    pointer-events: none;
}
.header-wrap::after {
    content: '';
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%);
    bottom: -60px; left: 15%;
    animation: spinBlob 12s linear infinite reverse;
    pointer-events: none;
}

@keyframes spinBlob {
    0%   { transform: rotate(0deg) translateX(30px); }
    100% { transform: rotate(360deg) translateX(30px); }
}

.header-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 16px rgba(0,0,0,0.4);
    position: relative; z-index: 1;
}
.header-title span { color: #7fe8f0; }
.header-subtitle {
    margin-top: 0.5rem;
    font-size: 1rem;
    font-weight: 400;
    color: rgba(255,255,255,0.72);
    font-family: 'JetBrains Mono', monospace;
    position: relative; z-index: 1;
    letter-spacing: 0.04em;
}
.header-badge {
    display: inline-block;
    margin-top: 1rem;
    padding: 4px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #c8f8ff;
    letter-spacing: 0.06em;
    position: relative; z-index: 1;
}

/* ── SECTION TITLES ── */
.section-label {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #0d9eaf;
    margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 18px; height: 2px;
    background: #0d9eaf;
    border-radius: 2px;
}

/* ── CARD ── */
.card {
    background: rgba(10, 22, 45, 0.85);
    border: 1px solid rgba(13, 158, 175, 0.22);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(13, 158, 175, 0.45); }

/* ── Streamlit widgets styling ── */
[data-testid="stNumberInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(5, 13, 26, 0.9) !important;
    border: 1px solid rgba(13, 158, 175, 0.3) !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stFileUploader"] {
    background: rgba(5, 13, 26, 0.6) !important;
    border: 1.5px dashed rgba(13, 158, 175, 0.4) !important;
    border-radius: 12px !important;
}

label, .stLabel { color: #8ab4d0 !important; font-size: 0.85rem !important; }

/* ── CTA BUTTON ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7b1a2a 0%, #9e2a3a 40%, #1a6b7a 80%, #0d9eaf 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2.8rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 24px rgba(13,158,175,0.35) !important;
    transition: all 0.25s !important;
    width: 100%;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(13,158,175,0.55) !important;
    filter: brightness(1.1) !important;
}

/* ── METRIC CARD ── */
.metric-card {
    background: linear-gradient(135deg, rgba(123,26,42,0.25), rgba(13,158,175,0.18));
    border: 1px solid rgba(13,158,175,0.3);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.metric-value {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #e85070, #0d9eaf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-label {
    margin-top: 0.5rem;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7fa8c0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(13,158,175,0.4), transparent);
    margin: 2rem 0;
}

/* ── PLOTLY container ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Input label color override ── */
.stTimeInput label, .stNumberInput label,
.stFileUploader label, .stSelectbox label {
    color: #8ab4d0 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050d1a; }
::-webkit-scrollbar-thumb { background: rgba(13,158,175,0.4); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-wrap">
    <div class="header-title">Opti<span>Border</span></div>
    <div class="header-subtitle">customs workforce optimization · algorithm R₀</div>
    <div class="header-badge">▸ portfolio project #01</div>
</div>
""",
    unsafe_allow_html=True,
)


# ── INPUTS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Service target</div>', unsafe_allow_html=True
    )
    target_wait = st.number_input(
        "Target average wait time (min)",
        min_value=1.0,
        max_value=120.0,
        value=10.0,
        step=0.5,
        format="%.1f",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Passenger data</div>', unsafe_allow_html=True
    )
    arrivals_df = st.file_uploader(
        "Upload Excel file of arrivals (.xlsx)",
        type=["xlsx", "xls"],
        help="One column of passenger arrivals per 15-min slot",
    )
    average_df = st.file_uploader(
        "Upload Excel file  of average time(.xlsx)",
        type=["xlsx", "xls"],
        help="One column of passenger arrivals per 15-min slot",
    )
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    cN = st.number_input(
        "Coût marginal d'un douanier :",
        min_value=1.0,
        max_value=8000.0,
        value=2000.0,
        step=0.5,
        format="%.1f",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    cT = st.number_input(
        "Coût marginal d'une minute supplémentaire",
        min_value=1.0,
        max_value=2000.0,
        value=2.0,
        step=0.5,
        format="%.1f",
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
if arrivals_df is None and average_df is None:
    arrivals_df = (
        r"C:\Users\Elfamoso\OneDrive\Bureau\Veille\Portfolio\Customs\Arrivals_HD.xlsx"
    )
    average_df = (
        r"C:\Users\Elfamoso\OneDrive\Bureau\Veille\Portfolio\Customs\Average_time.xlsx"
    )
esc = E0(arrivals_df, average_df, target_wait, cN, cT)


# ── RUN BUTTON ─────────────────────────────────────────────────────────────────
run = st.button("⚙  Run optimization · R₀")

if run:
    esc.print_Q()
    esc.print_N()
    esc.get_results()
