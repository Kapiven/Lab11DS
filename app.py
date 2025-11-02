import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="Dashboard Tweets - Desastres", layout="wide")

#Palette / Style
PRIMARY_RED = "#C62828"    # disaster
PRIMARY_BLUE = "#0277BD"   # non-disaster
ACCENT = "#FFC107"
BG = "#F5F5F7"
TEXT = "#1C1C1E"

st.markdown(f"""
    <style>
        .stApp {{ background-color: {TEXT}; color: {BG}; }}
        .title {{ font-weight:700; }}
    </style>
""", unsafe_allow_html=True)

#Utilities & Caching
@st.cache_data(show_spinner=False)
def load_data(path="tweets_clean_sentiment.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].fillna("").astype(str)
    else:
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
   
    if "sentiment" not in df.columns:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            df["sentiment"] = df["text"].fillna("").astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
        except Exception:
            df["sentiment"] = 0.0
    df["target"] = df["target"].fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def fit_vectorizer(corpus, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(corpus)
    return tfidf, X

@st.cache_data(show_spinner=True)
def train_models(_X, _y):
    # Split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }
    trained = {}
    metrics = {}
    probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        trained[name] = model
        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}
    
        try:
            prob = model.predict_proba(X_test)[:, 1]
            probs[name] = (prob, y_test)
        except Exception:
            probs[name] = (None, y_test)
    return trained, metrics, probs

#Load data
with st.spinner("Cargando datos..."):
    try:
        df = load_data("tweets_clean_sentiment.csv")
    except FileNotFoundError:
        st.error("Archivo 'tweets_clean_sentiment.csv' no encontrado en la carpeta. Genera el CSV según instrucciones previas.")
        st.stop()

#Sidebar: Controls
st.sidebar.title("Filtros y Controles")
st.sidebar.markdown("**Explora los datos y modelos**")

cat_option = st.sidebar.selectbox("Filtrar por categoría:", ("Ambos", "Desastres (1)", "No desastres (0)"))
if cat_option == "Desastres (1)":
    df_filtered = df[df["target"] == 1].copy()
elif cat_option == "No desastres (0)":
    df_filtered = df[df["target"] == 0].copy()
else:
    df_filtered = df.copy()

sent_min, sent_max = st.sidebar.slider("Rango de sentimiento (compound VADER)", -1.0, 1.0, (-1.0, 1.0), step=0.01)
df_filtered = df_filtered[(df_filtered["sentiment"] >= sent_min) & (df_filtered["sentiment"] <= sent_max)]

top_n = st.sidebar.slider("Top N palabras/bigramas", 5, 60, 20)
detail_level = st.sidebar.radio("Nivel de detalle (detalle UI)", ("Básico", "Avanzado"))
selected_models = st.sidebar.multiselect("Modelos para comparar (elige 1-3)", ["Logistic Regression", "Naive Bayes", "Random Forest"], default=["Logistic Regression","Naive Bayes","Random Forest"])

#Interactive selected word (populated from top words)
#We'll compute top words below and show selectbox dynamically in main area (so it updates with filters)

st.sidebar.markdown("---")
st.sidebar.markdown("**UX tips:** Usa el filtro de sentimiento para reducir ruido. Selecciona palabras para ver tweets que las contienen.")
st.sidebar.markdown("Paleta: 🔴=Desastre  🔵=No desastre  🟡=Acciones")

st.title("Dashboard Interactivo — Tweets sobre Desastres")
st.markdown("**Objetivo:** Explorar lenguaje, sentimiento y desempeño de modelos para detectar desastres en Twitter.")

# Row 1: Class distribution + Sentiment histogram + Wordcloud
c1, c2, c3 = st.columns([1,1,1])

with c1:
    st.subheader("Distribución de clases")
    counts = df_filtered["target"].value_counts().reindex([1,0]).fillna(0)
    labels = ["Desastre (1)", "No desastre (0)"]
    fig_pie = px.pie(
        names=labels,
        values=[counts.get(1,0), counts.get(0,0)],
        hole=0.45,
        color_discrete_sequence=[PRIMARY_RED, PRIMARY_BLUE]
    )
    st.plotly_chart(fig_pie, width='stretch')

with c2:
    st.subheader("Distribución de Sentimiento")
    fig_sent = px.histogram(df_filtered, x="sentiment", nbins=30, marginal="box",
                            color_discrete_sequence=[PRIMARY_RED if cat_option!="No desastres (0)" else PRIMARY_BLUE])
    st.plotly_chart(fig_sent, width='stretch')

with c3:
    st.subheader("Nube de palabras")
    text = " ".join(df_filtered["clean_text"].fillna("").astype(str))
    if len(text.strip()) == 0:
        st.info("No hay texto para generar la nube con los filtros actuales.")
    else:
        wc = WordCloud(width=900, height=400, background_color="white").generate(text)
        fig_wc, ax = plt.subplots(figsize=(9,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

st.markdown("---")

# Row 2: Top words (left) and top bigrams (right)
c4, c5 = st.columns(2)

with c4:
    st.subheader("Top de palabras")
    st.markdown("Click o selecciona una palabra abajo")
    cv = CountVectorizer(ngram_range=(1,1), stop_words="english")
    clean_series = df_filtered["clean_text"].fillna("").astype(str)
    Xc = cv.fit_transform(clean_series)
    words = cv.get_feature_names_out()
    freqs = Xc.toarray().sum(axis=0)
    freq_df = pd.DataFrame({"word": words, "count": freqs}).sort_values("count", ascending=False).head(top_n)
    fig_bar = px.bar(freq_df.sort_values("count"), x="count", y="word", orientation="h", title=f"Top {top_n} palabras",
                     color="count", color_continuous_scale="Reds" if cat_option!="No desastres (0)" else "Blues")
    st.plotly_chart(fig_bar, width='stretch')

with c5:
    st.subheader("Top bigramas")
    cv2 = CountVectorizer(ngram_range=(2,2), stop_words="english")
    Xb = cv2.fit_transform(clean_series)
    bigrams = cv2.get_feature_names_out()
    bf = Xb.toarray().sum(axis=0)
    bigram_df = pd.DataFrame({"bigram": bigrams, "count": bf}).sort_values("count", ascending=False).head(top_n)
    fig_bi = px.bar(bigram_df.sort_values("count"), x="count", y="bigram", orientation="h", title=f"Top {top_n} bigramas",
                    color="count", color_continuous_scale="Oranges")
    st.plotly_chart(fig_bi, width='stretch')

# Word selection
word_options = list(freq_df["word"].values)
selected_word = st.selectbox("Selecciona una palabra para filtrar tweets (enlaza con tabla de tweets)", options=["(ninguna)"] + word_options)

# Row 3: Co-occurrence heatmap + sample tweets table
c6, c7 = st.columns([1.2, 1])

with c6:
    st.subheader("Matriz de co-ocurrencia (top palabras)")
    
    top_k = min(20, len(freq_df))
    top_words_k = list(freq_df["word"].head(top_k))
  
    from sklearn.feature_extraction.text import CountVectorizer as CV2
    vec = CV2(vocabulary=top_words_k, binary=True)
    M = vec.fit_transform(clean_series).toarray()
    co = np.dot(M.T, M)
    fig_heat = px.imshow(co, x=top_words_k, y=top_words_k, labels=dict(x="word", y="word", color="count"),
                        color_continuous_scale="Blues")
    st.plotly_chart(fig_heat, width='stretch')

with c7:
    st.subheader("Ejemplos de tweets (filtrados)")
    if selected_word != "(ninguna)":
        subset = df_filtered[df_filtered["clean_text"].str.contains(rf"\b{selected_word}\b", na=False, case=False)]
    else:
        subset = df_filtered
    # show top 10 by negativity 
    subset_show = subset.sort_values("sentiment").head(15)[["text", "clean_text", "sentiment", "target"]]
    st.dataframe(subset_show.reset_index(drop=True), use_container_width=True)

st.markdown("---")

#Models: vectorize & train
st.subheader("Modelos y desempeño")
st.markdown("Se entrenan modelos simples (Logistic Regression, Naive Bayes, Random Forest) sobre TF-IDF del texto. La tabla y gráficas permiten comparar métricas y ver matrices de confusión.")

# Vectorize full dataset (use TF-IDF)
tfidf, X = fit_vectorizer(df["clean_text"].fillna("").astype(str), max_features=5000)
y = df["target"].astype(int)

with st.spinner("Entrenando modelos... (cacheado; demora 10-40s la primera vez)"):
    models_trained, metrics_dict, probs_dict = train_models(X, y)

# Model comparison table
comp_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
comp_df = comp_df[["accuracy", "precision", "recall", "f1"]].round(3)
comp_df = comp_df.reset_index().rename(columns={"index": "model"})
st.subheader("Tabla comparativa de modelos")
st.write("Selecciona qué modelos comparar (izquierda) y/o marca para ver su matriz de confusión (derecha).")
# multi-select for comparison
models_to_compare = st.multiselect("Modelos a mostrar en la tabla", options=list(metrics_dict.keys()), default=selected_models)
if len(models_to_compare) == 0:
    st.info("Selecciona al menos un modelo para ver la tabla comparativa.")
else:
    comp_view = comp_df[comp_df["model"].isin(models_to_compare)].set_index("model")
    st.table(comp_view.style.format("{:.3f}"))

# Bar chart of selected metrics
metric_choice = st.selectbox("Métrica a graficar:", ["accuracy", "precision", "recall", "f1"])
bar_plot = comp_df[comp_df["model"].isin(models_to_compare)].sort_values(metric_choice, ascending=False)
fig_m = px.bar(bar_plot, x="model", y=metric_choice, color="model", text=metric_choice,
               color_discrete_sequence=[PRIMARY_RED if "Logistic" in m else PRIMARY_BLUE if "Naive" in m else ACCENT for m in bar_plot["model"]])
st.plotly_chart(fig_m, width='stretch')

# Confusion matrix viewer
st.subheader("Matriz de confusión (modelo seleccionado)")
model_selected_for_cm = st.selectbox("Selecciona modelo para ver su matriz de confusión", options=list(metrics_dict.keys()))
cm = metrics_dict[model_selected_for_cm]["cm"]
fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicho", y="Real"))
st.plotly_chart(fig_cm, width='stretch')

# ROC curves (Avanzado)
if detail_level == "Avanzado":
    st.subheader("Curvas ROC (Avanzado)")
    fig_roc = go.Figure()
    for name, (prob, y_test) in probs_dict.items():
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {roc_auc:.2f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, width='stretch')

st.markdown("---")
# Feature: show learning curve stub (synthetic small plot) to satisfy "tasa de aprendizaje" requirement
st.subheader("Tasa de aprendizaje / comportamiento de validación (simplificado)")
st.markdown("Este panel muestra una curva simplificada (muestra cómo varía accuracy con más datos de entrenamiento).")
# Quick synthetic learning curve using incremental training sizes
sizes = [0.1, 0.2, 0.4, 0.6, 0.8]
lc_metrics = []
for s in sizes:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=s, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    lc_metrics.append(accuracy_score(y_te, m.predict(X_te)))
fig_lc = px.line(x=sizes, y=lc_metrics, markers=True, labels={"x":"Fracción de entrenamiento", "y":"Accuracy"})
st.plotly_chart(fig_lc, width='stretch')

st.markdown("---")
st.subheader("Acción y exportes")
st.markdown("Puedes exportar la tabla comparativa o los tweets filtrados para incluirlos en reportes.")
if st.button("Exportar tweets filtrados (CSV)"):
    st.download_button("Descargar CSV", data=df_filtered.to_csv(index=False).encode('utf-8'), file_name="tweets_filtrados.csv")

st.caption("Dashboard diseñado para audiencias técnicas y directivos — incluye explicaciones breves y controles claros para exploración.")