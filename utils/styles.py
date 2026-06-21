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
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        [data-testid="stSidebar"] {{
            background-color: {WC_COLORS['navy']} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        .main-header {{
            background-color: {WC_COLORS['maroon']};
            color: white !important;
            font-family: 'Trebuchet MS', sans-serif;
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            border-bottom: 5px solid {WC_COLORS['gold']};
            margin-bottom: 40px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}
        .stMetric {{
            background-color: white !important;
            padding: 20px !important;
            border-radius: 15px !important;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
            border-top: 4px solid {WC_COLORS['gold']} !important;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            color: {WC_COLORS['maroon']} !important;
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
