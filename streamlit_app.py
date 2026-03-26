import pandas as pd
import streamlit as st

st.set_page_config(page_title="Explorador de clusters", layout="wide")

@st.cache_data
def load_data():
    docs = pd.read_csv("data/clustered_docs.csv")
    words = pd.read_csv("data/cluster_top_words.csv")
    stats = pd.read_csv("data/cluster_stats.csv")
    return docs, words, stats

docs, words, stats = load_data()

st.title("Explorador interactivo de clusters")
st.write(
    "Aplicación para explorar clusters temáticos obtenidos a partir de abstracts científicos utilizando TF-IDF y KMeans."
)

clusters = sorted(docs["cluster"].dropna().unique().tolist())
selected_cluster = st.sidebar.selectbox("Seleccionar cluster", clusters)
n_examples = st.sidebar.slider("Cantidad de ejemplos", 3, 15, 5)

cluster_docs = docs[docs["cluster"] == selected_cluster]
cluster_words = words[words["cluster"] == selected_cluster].sort_values("count", ascending=False)
cluster_stats = stats[stats["cluster"] == selected_cluster]

st.subheader(f"Cluster {selected_cluster}")

col1, col2 = st.columns(2)

with col1:
    st.metric("Cantidad de documentos", len(cluster_docs))

with col2:
    if not cluster_stats.empty:
        st.metric(
            "Longitud promedio del abstract",
            round(cluster_stats["avg_abstract_length"].iloc[0], 2)
        )

st.markdown("### Palabras más frecuentes")
st.dataframe(cluster_words.head(15), use_container_width=True)

st.markdown("### Ejemplos de documentos")
st.dataframe(
    cluster_docs[["title", "journal", "abstract_clean"]].head(n_examples),
    use_container_width=True
)

st.markdown("### Journals más frecuentes")
journal_counts = (
    cluster_docs["journal"]
    .fillna("NULL")
    .value_counts()
    .reset_index()
)
journal_counts.columns = ["journal", "count"]
st.dataframe(journal_counts.head(10), use_container_width=True)