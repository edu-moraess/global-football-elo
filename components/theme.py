"""Theme v2 — Design system Bloomberg-dark."""

GOLD   = "#C8A951"
RED    = "#E84040"
GREEN  = "#2ECC71"
BLUE   = "#4A9EFF"
DARK   = "#080C10"
CARD   = "#0E1318"
BORDER = "#1E2A3A"
TEXT   = "#E8E8E8"
MUTED  = "#7A8A9A"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;600;700&display=swap');

html, body, .stApp {{
    background-color: {DARK} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: #080C12 !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {{
    color: {MUTED} !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 2px solid {GOLD};
    border-radius: 4px;
    padding: 14px 16px !important;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; }}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; }}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}}

h1 {{ color: {GOLD} !important; font-size: 1.4rem !important; font-weight: 700; letter-spacing: -0.02em; border-bottom: 1px solid {BORDER}; padding-bottom: 10px; }}
h2 {{ color: {TEXT} !important; font-size: 1.05rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {MUTED} !important; }}
h3 {{ color: {TEXT} !important; font-size: 0.95rem !important; font-weight: 600; }}

.stCaption {{ color: {MUTED} !important; font-size: 0.7rem; }}
hr {{ border-color: {BORDER} !important; }}

[data-testid="stSelectbox"] > div > div {{
    background: {CARD} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    font-size: 0.85rem;
}}

.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

[data-testid="stAlert"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px;
    font-size: 0.82rem;
}}

[data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER};
    gap: 0;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: {MUTED} !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 6px 14px !important;
    border-radius: 0 !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
}}

.stSpinner > div {{ border-top-color: {GOLD} !important; }}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def ticker_bar(items: list[tuple[str, str, str]]):
    import streamlit as st
    parts = []
    for label, value, delta in items:
        color = GREEN if delta.startswith('+') else (RED if delta.startswith('-') else MUTED)
        parts.append(
            f"<span style='color:{MUTED};font-size:0.7rem;text-transform:uppercase;"
            f"letter-spacing:.08em;margin-right:4px'>{label}</span>"
            f"<span style='color:{TEXT};font-family:JetBrains Mono,monospace;"
            f"font-size:0.82rem;margin-right:4px'>{value}</span>"
            f"<span style='color:{color};font-size:0.72rem;margin-right:20px'>{delta}</span>"
        )
    html = (
        f"<div style='background:{CARD};border:1px solid {BORDER};"
        f"border-radius:4px;padding:8px 14px;overflow-x:auto;white-space:nowrap;"
        f"display:flex;align-items:center;gap:0;margin-bottom:12px'>"
        + "".join(parts) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def stat_card(label: str, value: str, sub: str = "", accent: str = GOLD):
    import streamlit as st
    st.markdown(
        f"<div style='background:{CARD};border:1px solid {BORDER};"
        f"border-top:2px solid {accent};border-radius:4px;"
        f"padding:12px 14px;margin-bottom:8px'>"
        f"<div style='color:{MUTED};font-size:0.68rem;text-transform:uppercase;"
        f"letter-spacing:.1em;margin-bottom:4px'>{label}</div>"
        f"<div style='color:{TEXT};font-family:JetBrains Mono,monospace;"
        f"font-size:1.2rem;font-weight:600'>{value}</div>"
        f"{'<div style=color:' + MUTED + ';font-size:.7rem;margin-top:2px>' + sub + '</div>' if sub else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def section_header(title: str, subtitle: str = ""):
    import streamlit as st
    st.markdown(
        f"<div style='margin:18px 0 10px;'>"
        f"<span style='color:{GOLD};font-size:0.65rem;text-transform:uppercase;"
        f"letter-spacing:.14em;font-weight:600'>{subtitle}</span>"
        f"<h3 style='color:{TEXT};margin:2px 0 0;font-size:0.95rem;"
        f"font-weight:600;letter-spacing:-0.01em'>{title}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )


def narrative_box(title: str, body: str, accent: str = GOLD):
    import streamlit as st
    st.markdown(
        f"<div style='background:{CARD};border:1px solid {BORDER};"
        f"border-left:3px solid {accent};border-radius:4px;"
        f"padding:14px 16px;margin:8px 0'>"
        f"<div style='color:{accent};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:.1em;font-weight:600;margin-bottom:6px'>{title}</div>"
        f"<div style='color:{TEXT};font-size:0.85rem;line-height:1.6'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True
    )