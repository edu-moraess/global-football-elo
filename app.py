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

st.title("⚽ Global Football Quant Intelligence")
st.caption("Análise Quantitativa de Futebol Internacional")
st.divider()

st.markdown("""
O futebol não é apenas 22 homens correndo atrás de uma bola. É matemática, história, psicologia e geografia
condensados em 90 minutos. Este projeto usa o **motor Elo-Poisson-Dixon-Coles** para decifrar o jogo
como nunca antes: não para apostar, mas para **entender**.
""")

st.info("👈 Use a barra lateral para navegar entre as 8 seções analíticas.")

st.caption("© Global Football Quant Intelligence | Motor Elo-Poisson-Dixon-Coles")
