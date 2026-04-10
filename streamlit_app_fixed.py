import re
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(page_title="Explorador de clusters científicos", layout="wide")


def find_file(filename: str) -> Path:
    candidates = [
        Path("data") / filename,
        Path(filename),
        Path(".") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No se encontró '{filename}'. Busqué en: "
        + ", ".join(str(p) for p in candidates)
    )


@st.cache_data
def load_data():
    docs_path = find_file("clustered_docs.csv")
    stats_path = find_file("cluster_stats.csv")

    docs = pd.read_csv(docs_path)
    stats = pd.read_csv(stats_path)

    rename_map = {}
    if "prediction" in docs.columns and "cluster" not in docs.columns:
        rename_map["prediction"] = "cluster"
    if rename_map:
        docs = docs.rename(columns=rename_map)

    required_cols = {"title", "abstract_clean", "cluster"}
    missing = required_cols - set(docs.columns)
    if missing:
        raise ValueError(
            "Faltan columnas necesarias en clustered_docs.csv: "
            + ", ".join(sorted(missing))
        )

    docs["cluster"] = pd.to_numeric(docs["cluster"], errors="coerce")
    docs = docs.dropna(subset=["cluster"]).copy()
    docs["cluster"] = docs["cluster"].astype(int)

    if "abstract_length" in docs.columns:
        docs["abstract_length"] = pd.to_numeric(docs["abstract_length"], errors="coerce")

    if "publish_year" in docs.columns:
        docs["publish_year"] = pd.to_numeric(docs["publish_year"], errors="coerce")

    return docs, stats


@st.cache_data
def build_cluster_top_words(docs: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    stopwords = {
        "the", "and", "for", "with", "that", "from", "this", "were", "have", "has",
        "had", "into", "than", "then", "they", "their", "there", "after", "before",
        "using", "used", "use", "our", "can", "may", "also", "such", "these", "those",
        "are", "was", "been", "being", "but", "not", "all", "any", "each", "other",
        "more", "most", "some", "many", "few", "via", "per", "due", "new", "two",
        "one", "among", "over", "under", "between", "during", "within", "without",
        "covid", "19", "sars", "cov", "coronavirus", "study", "results", "conclusions",
        "background", "methods", "patients", "patient", "disease", "analysis",
        "data", "clinical", "infected", "infection"
    }

    rows = []
    for cluster_id in sorted(docs["cluster"].dropna().unique()):
        text_series = docs.loc[docs["cluster"] == cluster_id, "abstract_clean"].fillna("")
        tokens = []

        for text in text_series:
            found_words = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
            found_words = [w for w in found_words if w not in stopwords]
            tokens.extend(found_words)

        counts = Counter(tokens).most_common(top_n)
        for word, count in counts:
            rows.append({"cluster": cluster_id, "word": word, "count": count})

    return pd.DataFrame(rows)


@st.cache_data
def build_cluster_labels(words: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster_id in sorted(words["cluster"].dropna().unique()):
        cluster_words = (
            words[words["cluster"] == cluster_id]
            .sort_values("count", ascending=False)["word"]
            .head(6)
            .tolist()
        )

        top_text = ", ".join(cluster_words[:4])

        if any(w in cluster_words for w in ["vaccine", "vaccination", "immune", "antibody"]):
            label = "Vaccines and Immunology"
        elif any(w in cluster_words for w in ["trial", "treatment", "therapy", "drug"]):
            label = "Treatment and Clinical Trials"
        elif any(w in cluster_words for w in ["public", "health", "pandemic", "policy"]):
            label = "Public Health and Pandemic Response"
        elif any(w in cluster_words for w in ["virus", "viral", "host", "cell"]):
            label = "Virology and Host Response"
        elif any(w in cluster_words for w in ["risk", "mortality", "severity", "outcome"]):
            label = "Risk Factors and Outcomes"
        elif any(w in cluster_words for w in ["care", "hospital", "group", "patient"]):
            label = "Healthcare and Patient Care"
        else:
            label = f"Topic around {cluster_words[0].title()}" if cluster_words else "Unlabeled Topic"

        explanation = (
            f"This cluster appears to represent research related to: {top_text}."
            if cluster_words else
            "This cluster could not be automatically described."
        )

        rows.append({
            "cluster": cluster_id,
            "cluster_label": label,
            "cluster_explanation": explanation
        })

    return pd.DataFrame(rows)


@st.cache_data
def build_representative_docs(docs: pd.DataFrame, words: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    rows = []

    for cluster_id in sorted(docs["cluster"].dropna().unique()):
        cluster_docs = docs[docs["cluster"] == cluster_id].copy()
        top_words = (
            words[words["cluster"] == cluster_id]
            .sort_values("count", ascending=False)["word"]
            .head(10)
            .tolist()
        )

        def score_text(text):
            text = str(text).lower()
            return sum(1 for w in top_words if w in text)

        cluster_docs["representative_score"] = cluster_docs["abstract_clean"].fillna("").apply(score_text)

        top_docs = cluster_docs.sort_values(
            ["representative_score"],
            ascending=False
        ).head(top_k)

        rows.append(top_docs)

    if rows:
        return pd.concat(rows, ignore_index=True)

    return pd.DataFrame()


def main():
    try:
        docs, stats = load_data()
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        st.stop()

    words = build_cluster_top_words(docs)
    cluster_labels = build_cluster_labels(words)
    representative_docs = build_representative_docs(docs, words, top_k=10)

    st.title("Explorador interactivo de clusters científicos")
    st.write(
        "Aplicación para explorar clusters temáticos obtenidos a partir de abstracts "
        "científicos usando PySpark, TF-IDF y KMeans."
    )

    with st.expander("Metodología"):
        st.write("""
        Los documentos fueron preprocesados y representados con TF-IDF.
        Luego se aplicó KMeans para identificar grupos temáticos.
        Esta app permite explorar cada cluster a través de sus términos,
        documentos representativos y journals.
        """)

    st.sidebar.header("Filtros")

    clusters = sorted(docs["cluster"].dropna().unique().tolist())

    journal_options = ["Todos"]
    if "journal" in docs.columns:
        journal_counts_all = docs["journal"].fillna("Sin journal").value_counts()
        valid_journals = journal_counts_all[journal_counts_all > 2].index.tolist()
        journal_options = ["Todos"] + sorted(valid_journals)

    with st.sidebar.form("filtros_form"):
        selected_cluster = st.selectbox("Seleccionar cluster", clusters)
        n_examples = st.slider("Cantidad de ejemplos", 3, 15, 5)
        search_term = st.text_input("Buscar palabra en abstracts")
        selected_journal = st.selectbox("Filtrar por journal", journal_options)
        st.form_submit_button("Aplicar filtros")

    cluster_docs = docs[docs["cluster"] == selected_cluster].copy()

    if search_term:
        cluster_docs = cluster_docs[
            cluster_docs["abstract_clean"].fillna("").str.contains(search_term, case=False, na=False)
        ]

    if selected_journal != "Todos" and "journal" in cluster_docs.columns:
        cluster_docs = cluster_docs[cluster_docs["journal"] == selected_journal]

    cluster_words = words[words["cluster"] == selected_cluster].sort_values("count", ascending=False)
    cluster_stats = stats[stats["cluster"] == selected_cluster]
    cluster_label_row = cluster_labels[cluster_labels["cluster"] == selected_cluster]

    if not cluster_label_row.empty:
        cluster_name = cluster_label_row["cluster_label"].iloc[0]
    else:
        cluster_name = f"Cluster {selected_cluster}"

    st.subheader(f"Cluster {selected_cluster} — {cluster_name}")

    if not cluster_label_row.empty:
        st.info(cluster_label_row["cluster_explanation"].iloc[0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cantidad de documentos", len(cluster_docs))

    with col2:
        if not cluster_stats.empty and "avg_abstract_length" in cluster_stats.columns:
            st.metric(
                "Longitud promedio del abstract",
                round(float(cluster_stats["avg_abstract_length"].iloc[0]), 2)
            )
        else:
            st.metric("Longitud promedio del abstract", "N/D")

    with col3:
        if not cluster_label_row.empty:
            st.metric("Tema estimado", cluster_label_row["cluster_label"].iloc[0])
        else:
            st.metric("Tema estimado", "N/D")

    if not cluster_words.empty:
        st.caption("Top signal words: " + ", ".join(cluster_words["word"].head(5).tolist()))

    tab1, tab2, tab3 = st.tabs([
        "Señales léxicas",
        "Documentos representativos",
        "Distribución por journal"
    ])

    with tab1:
        st.markdown("### Palabras más frecuentes")
        if cluster_words.empty:
            st.info("No hay palabras para mostrar en este cluster.")
        else:
            top_n_words = st.slider("Cantidad de palabras a mostrar", 5, 20, 10)

            plot_df = cluster_words.head(top_n_words).sort_values("count", ascending=True)

            fig = px.bar(
                plot_df,
                x="count",
                y="word",
                orientation="h",
                title=f"Top palabras del cluster {selected_cluster}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Documentos más representativos")

        sample_docs = representative_docs[
            representative_docs["cluster"] == selected_cluster
        ].copy()

        if search_term:
            sample_docs = sample_docs[
                sample_docs["abstract_clean"].fillna("").str.contains(search_term, case=False, na=False)
            ]

        if selected_journal != "Todos" and "journal" in sample_docs.columns:
            sample_docs = sample_docs[sample_docs["journal"] == selected_journal]

        sample_docs = sample_docs.head(n_examples)

        if sample_docs.empty:
            st.info("No hay documentos para mostrar con los filtros seleccionados.")
        else:
            for _, row in sample_docs.iterrows():
                title = row["title"] if pd.notna(row["title"]) else "Sin título"
                st.markdown(f"**{title}**")

                if "journal" in row and pd.notna(row["journal"]):
                    st.caption(f"Journal: {row['journal']}")

                abstract = str(row["abstract_clean"]) if pd.notna(row["abstract_clean"]) else ""
                short_abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                st.write(short_abstract)

                if "representative_score" in row:
                    st.caption(f"Representative score: {row['representative_score']}")

                with st.expander("Ver abstract completo"):
                    st.write(abstract)

                st.markdown("---")

    with tab3:
        st.markdown("### Journals más frecuentes")
        if "journal" in cluster_docs.columns:
            journal_counts = (
                cluster_docs["journal"]
                .fillna("Sin journal")
                .value_counts()
                .head(10)
                .reset_index()
            )
            journal_counts.columns = ["journal", "count"]

            fig = px.bar(
                journal_counts.sort_values("count", ascending=True),
                x="count",
                y="journal",
                orientation="h",
                title=f"Top journals del cluster {selected_cluster}"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(journal_counts, use_container_width=True)
        else:
            st.info("No hay columna 'journal' en los datos exportados.")

    st.markdown("---")
    st.markdown(
        "La app usa directamente los resultados exportados por el notebook: "
        "`clustered_docs.csv` y `cluster_stats.csv`."
    )


if __name__ == "__main__":
    main()
