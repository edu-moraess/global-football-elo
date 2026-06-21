# utils/styles.py
import streamlit as st

WC_COLORS = {
    "gold": "#D4AF37",
    "maroon": "#8A1538",
    "white": "#FFFFFF",
    "navy": "#001F3F",
    "green": "#006233",
    "bg": "#F8F9FA"
}

def apply_custom_styles():
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {WC_COLORS['bg']};
        }}
        .main-header {{
            color: {WC_COLORS['maroon']};
            font-family: 'Trebuchet MS', sans-serif;
            text-align: center;
            padding: 20px;
            border-bottom: 3px solid {WC_COLORS['gold']};
            margin-bottom: 30px;
        }}
        .stMetric {{
            background-color: white !important;
            padding: 15px !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            border-left: 5px solid {WC_COLORS['gold']} !important;
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid {WC_COLORS['gold']};
            border-radius: 10px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: #FFFFFF;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {WC_COLORS['maroon']} !important;
            color: white !important;
        }}
        </style>
        """, unsafe_allow_html=True)
