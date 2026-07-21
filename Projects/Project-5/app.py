"""
Project 5 — NLP Nearest-Neighbor Explorer
Find people whose biographies are most similar, using sentence-embedding meaning vectors.

Data (precomputed) is hosted on Hugging Face: RaayGunz/pinky-nlp-embeddings
  - pinky_embeddings.npy   : (42786, 384) meaning vectors, one per person
  - pinky_people.parquet   : names + biography text

Two search paths:
  - FAST PATH : typed name matches an existing person -> use their precomputed vector (instant)
  - SLOW PATH : no match -> load the model and embed the typed text live (works for any input)
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="NLP Neighbor Explorer", layout="wide")
st.title("NLP Nearest-Neighbor Explorer")
st.write(
    "Enter a famous person's name. The app finds the 10 people whose biographies "
    "are closest in *meaning*, using sentence-embedding vectors."
)

REPO_ID = "RaayGunz/pinky-nlp-embeddings"

# ----------------------------------------------------------------------
# Cached loaders — run ONCE, reused across interactions
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """Download the precomputed embeddings + people table from Hugging Face."""
    emb_path = hf_hub_download(repo_id=REPO_ID, filename="pinky_embeddings.npy", repo_type="dataset")
    ppl_path = hf_hub_download(repo_id=REPO_ID, filename="pinky_people.parquet", repo_type="dataset")
    embeddings = np.load(emb_path)
    people = pd.read_parquet(ppl_path)
    return embeddings, people


@st.cache_resource
def load_model():
    """Load the SentenceTransformer model — only called on the slow path (new names)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


embeddings, people = load_data()

# ----------------------------------------------------------------------
# Core: given a query vector, return the top-k most similar people
# ----------------------------------------------------------------------
def nearest(query_vec, k=10, exclude_idx=None):
    """Cosine-similarity the query against all embeddings; return top-k (name, score)."""
    sims = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    order = sims.argsort()[::-1]
    results = []
    for idx in order:
        if idx == exclude_idx:      # skip the person themselves (self-match)
            continue
        results.append((people.iloc[idx]["name"], sims[idx]))
        if len(results) == k:
            break
    return results

# ----------------------------------------------------------------------
# User input
# ----------------------------------------------------------------------
query = st.text_input("Enter a name:", "Pinky Lai")

if query:
    # Case-insensitive match against existing people
    mask = people["name"].str.contains(query, case=False, na=False)
    matches = people[mask]

    if len(matches) == 0:
        # ---------- SLOW PATH: name not in dataset, embed live ----------
        st.info(f"'{query}' is not in the dataset — embedding your text live to find neighbors.")
        model = load_model()
        query_vec = model.encode([query])[0]
        results = nearest(query_vec, k=10, exclude_idx=None)
        chosen_name, chosen_text = query, None

    elif len(matches) > 1:
        # ---------- Ambiguous: let the user pick ----------
        choice = st.selectbox("Multiple matches — which did you mean?", matches["name"].tolist())
        row_idx = people.index[people["name"] == choice][0]
        query_vec = embeddings[row_idx]
        results = nearest(query_vec, k=10, exclude_idx=row_idx)
        chosen_name = choice
        chosen_text = people.loc[row_idx, "text"]

    else:
        # ---------- FAST PATH: exactly one match, use precomputed vector ----------
        row_idx = matches.index[0]
        query_vec = embeddings[row_idx]
        results = nearest(query_vec, k=10, exclude_idx=row_idx)
        chosen_name = matches.iloc[0]["name"]
        chosen_text = people.loc[row_idx, "text"]

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    st.subheader(f"10 closest to: {chosen_name}")

    res_df = pd.DataFrame(results, columns=["Name", "Similarity"])
    res_df["Similarity"] = res_df["Similarity"].round(3)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(res_df, use_container_width=True)
        # bar chart of similarity magnitude
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(res_df["Name"][::-1], res_df["Similarity"][::-1], color="steelblue")
        ax.set_xlabel("Cosine similarity")
        ax.set_title("Neighbor magnitude")
        st.pyplot(fig)

    with col2:
        # Word cloud + sentiment, only if we have the person's text
        if chosen_text:
            wc = WordCloud(width=600, height=400, background_color="white",
                           stopwords={"the", "a", "an", "and", "of", "in", "is", "was", "to"}
                           ).generate(chosen_text)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            ax2.set_title(f"Word cloud — {chosen_name}")
            st.pyplot(fig2)

            blob = TextBlob(chosen_text)
            st.metric("Polarity", round(blob.sentiment.polarity, 3))
            st.metric("Subjectivity", round(blob.sentiment.subjectivity, 3))
        else:
            st.write("(Word cloud & sentiment shown for people in the dataset.)")
