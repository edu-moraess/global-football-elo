import streamlit as st

st.set_page_config(
    page_title="Global Football Quant Intelligence",
    layout="wide",
    page_icon="⚽"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    h1, h2, h3, h4 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; margin-bottom:8px;'>⚽ Global Football Quant Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888; font-size:1.05em;'>Análise Quantitativa de Futebol Internacional</p>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 24px; border-radius: 12px; margin-bottom: 20px;">
<p style="color: #ccc; line-height: 1.8; margin: 0;">
O futebol não é apenas 22 homens correndo atrás de uma bola. É matemática, história, psicologia e geografia 
condensados em 90 minutos. Este projeto usa o <b>motor Elo-Poisson-Dixon-Coles</b> para decifrar o jogo 
como nunca antes: não para apostar, mas para <b>entender</b>.
</p>
</div>
""", unsafe_allow_html=True)

st.info("""
👈 Use a <b>barra lateral</b> para navegar entre as seções analíticas.

Cada página oferece uma lente diferente para o mesmo universo: seleções, probabilidades, histórias e cenários.
""")

st.caption("© Global Football Quant Intelligence | Motor Elo-Poisson-Dixon-Coles")
