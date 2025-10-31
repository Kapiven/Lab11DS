# Importación de librerías

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Carga del dataset original
df = pd.read_csv("train2.csv")

stop_words = set([
    # stopwords básicas
    "the","a","an","in","on","and","or","but","if","at","by","for","with",
    "about","against","between","into","through","during","before","after",
    "to","from","up","then","once","here","there","when","where","why","how",
    "all","any","both","each","few","more","most","other","some","such","no",
    "nor","not","only","own","same","so","than","too","very","is","are","was",
    "were","be","been","being","of","do","does","did","doing","would","could",
    "should","can","will",

    # pronombres
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",

    # palabras de twitter / conversación
    "amp","rt","im","dont","cant","didnt","doesnt","youre","youve","ive","id",
    "ill","hes","shes","theyre","weve","lets","lol","omg","ugh","got","like",
    "just","know","time","new","day","love","people","going","good","think",
    "want","really","one"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Manejo de números: conservar 911 y algunos con palabras relevantes
    tokens = text.split()
    clean_tokens = []
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            if tok == "911":
                clean_tokens.append(tok)
            elif i+1 < len(tokens) and tokens[i+1] in ["dead", "injured", "wounded", "killed"]:
                clean_tokens.append(tok)
            elif i > 0 and tokens[i-1] == "magnitude":
                clean_tokens.append(tok)
        else:
            clean_tokens.append(tok)

    # Quitar stopwords y palabras de 1 caracter
    clean_tokens = [w for w in clean_tokens if w not in stop_words and len(w) > 1]

    return " ".join(clean_tokens)

df["clean_text"] = df["text"].apply(clean_text)
df[["text", "clean_text"]].head(10)

# ===== Añadir columna de sentimiento =====
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# ===== Guardar dataset limpio =====
df.to_csv("tweets_clean_sentiment.csv", index=False)

print("✅ Dataset listo: tweets_clean_sentiment.csv")
