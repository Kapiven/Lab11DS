import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# CONFIGURACIÓN DE ESTILO
st.set_page_config(page_title="Dashboard Tweets Desastres", layout="wide")

PRIMARY_RED = "#C62828"
PRIMARY_BLUE = "#0277BD"
ACCENT = "#FFC107"
BACKGROUND = "#F5F5F7"

st.markdown(f"""
    <style>
        body {{
            background-color: {BACKGROUND};
        }}
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv("tweets_clean_sentiment.csv")

# Renombrar por simplicitud
df = df.rename(columns={"target": "categoria"})

# UI PRINCIPAL 
st.title("Dashboard de Tweets sobre Desastres")
st.write("Análisis interactivo para detección de desastres reales en redes sociales.")

# FILTROS
col1, col2, col3 = st.columns(3)

with col1:
    categoria = st.selectbox("Filtrar por categoría:", ["Desastres", "No desastres"])
    cat_value = 1 if categoria == "Desastres" else 0
    df_filtered = df[df["categoria"] == cat_value]

with col2:
    top_n = st.slider("Número de palabras más frecuentes", 10, 50, 20)

with col3:
    sent_range = st.slider("Rango de sentimiento", -1.0, 1.0, (-1.0, 1.0))
    df_filtered = df_filtered[(df_filtered["sentiment"] >= sent_range[0]) & (df_filtered["sentiment"] <= sent_range[1])]

# VISUALIZACIÓN 1: NUBE DE PALABRAS 
st.subheader("Nube de Palabras")

text = " ".join(df_filtered["clean_text"].fillna("").astype(str))
wordcloud = WordCloud(width=900, height=400, background_color="white").generate(text)

fig_wc, ax = plt.subplots(figsize=(9,4))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)

# VISUALIZACIÓN 2: HISTOGRAMA DE SENTIMIENTO 
st.subheader("Distribución del Sentimiento")
fig_sent = px.histogram(
    df_filtered,
    x="sentiment",
    nbins=30,
    color_discrete_sequence=[PRIMARY_RED if cat_value == 1 else PRIMARY_BLUE]
)
st.plotly_chart(fig_sent, width='stretch')

# VISUALIZACIÓN 3: FRECUENCIA DE PALABRAS 
st.subheader("Palabras Más Frecuentes")

from sklearn.feature_extraction.text import CountVectorizer

# Convertir a texto y eliminar NaN antes de vectorizar
clean_series = df_filtered["clean_text"].fillna("").astype(str)

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(clean_series)
word_counts = X.toarray().sum(axis=0)
words = cv.get_feature_names_out()

freq_df = pd.DataFrame({"word": words, "count": word_counts})
freq_df = freq_df.sort_values("count", ascending=False).head(top_n)

fig_bar = px.bar(freq_df, x="count", y="word", orientation="h",
                 color="count", color_continuous_scale="Reds" if cat_value==1 else "Blues")
st.plotly_chart(fig_bar, width='stretch')

# VISUALIZACIÓN 4: COMPARACIÓN DE MODELOS
st.subheader("Comparación de Rendimiento de Modelos")

model_scores = pd.DataFrame({
    "Modelo": ["Regresión Logística", "Naive Bayes", "Random Forest"],
    "Accuracy (%)": [81, 79, 78]
})

fig_models = px.bar(model_scores, x="Modelo", y="Accuracy (%)", color="Modelo",
                    color_discrete_sequence=[PRIMARY_RED, PRIMARY_BLUE, ACCENT])
st.plotly_chart(fig_models, width='stretch')
