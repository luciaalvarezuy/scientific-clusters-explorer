import re
import joblib
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px

MANUAL_CLUSTER_LABELS = {
    0: "COVID-19 Clinical and Biomedical Research",
    1: "Hospital Care and COVID-19 Risk Factors",
    2: "Respiratory Complications and Health Systems",
    3: "Social, Behavioral and Food-related Impacts",
    4: "Genomic and Computational Virology"
}

st.set_page_config(page_title="Scientific Clusters Explorer", layout="wide")


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

        if cluster_id in MANUAL_CLUSTER_LABELS:
            label = MANUAL_CLUSTER_LABELS[cluster_id]
        elif any(w in cluster_words for w in ["vaccine", "vaccination", "immune", "antibody"]):
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


@st.cache_resource
def load_classifier():
    vectorizer_path = find_file("tfidf_vectorizer.joblib")
    classifier_path = find_file("cluster_classifier.joblib")

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    return vectorizer, classifier


def predict_text(text: str, vectorizer, classifier):
    X = vectorizer.transform([text])
    pred_cluster = int(classifier.predict(X)[0])

    confidence = None
    top_predictions = None

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        confidence = float(probs.max())

        classes = classifier.classes_
        pred_df = pd.DataFrame({
            "cluster": classes,
            "probability": probs
        }).sort_values("probability", ascending=False)

        top_predictions = pred_df.head(3).copy()
        top_predictions["cluster"] = top_predictions["cluster"].astype(int)

    return pred_cluster, confidence, top_predictions


def main():
    try:
        docs, stats = load_data()
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        st.stop()

    words = build_cluster_top_words(docs)
    cluster_labels = build_cluster_labels(words)
    representative_docs = build_representative_docs(docs, words, top_k=10)

    try:
        vectorizer, classifier = load_classifier()
        classifier_ready = True
    except Exception:
        classifier_ready = False

    st.title("Scientific Clusters Explorer")
    st.write(
        "Interactive NLP application for exploring thematic clusters of scientific abstracts "
        "and classifying new input text in real time. The app combines unsupervised topic discovery "
        "(KMeans over TF-IDF features) with a lightweight supervised classifier trained on cluster labels "
        "to support live inference."
    )

    with st.expander("Methodology"):
        st.write("""
        This application combines two complementary NLP workflows:
    
        **1. Cluster exploration**
        - Scientific abstracts were preprocessed and represented using TF-IDF
        - KMeans was applied to identify thematic clusters
        - Each cluster is described using its most frequent terms and representative documents
    
        **2. Live text classification**
        - A lightweight supervised classifier was trained using the existing cluster assignments as pseudo-labels
        - When a user pastes a new abstract, the app vectorizes the text with the saved TF-IDF model
        - The classifier then predicts the most likely topic/cluster and returns a confidence score when available
    
        **Language note**
        - The classifier was trained on scientific abstracts written mainly in English
        - It works best with English abstracts
        - Results for Spanish or other languages may be less reliable
        """)

    st.markdown("---")
    st.markdown("## Live Text Classifier")

    st.markdown(
    """
    Paste a scientific abstract or short research text and the model will predict the most likely topic.

    **How it works**
    - the input text is transformed using the saved TF-IDF vectorizer
    - a supervised classifier predicts the most likely cluster/topic
    - the app returns the predicted topic, the cluster ID, and confidence when available

    **Important**
    - the classifier was trained mainly on English scientific abstracts
    - performance is expected to be best for English input
    - predictions for Spanish or other languages may be less reliable
    """
)
    if classifier_ready:
        st.info(
            "For best performance, use English scientific abstracts. "
            "The model was trained mainly on English-language text."
        )
        user_input = st.text_area(
            "Paste an abstract or short scientific text",
            height=180,
           placeholder="Example (recommended in English): Patients with COVID-19 were analyzed to evaluate risk factors, outcomes, and public health implications..."
        )

        if st.button("Predict topic"):
            if user_input.strip():
                pred_cluster, confidence, top_predictions = predict_text(user_input, vectorizer, classifier)

                pred_label_row = cluster_labels[cluster_labels["cluster"] == pred_cluster]

                if not pred_label_row.empty:
                    pred_label = pred_label_row["cluster_label"].iloc[0]
                    pred_explanation = pred_label_row["cluster_explanation"].iloc[0]
                else:
                    pred_label = f"Cluster {pred_cluster}"
                    pred_explanation = "No explanation available."

                col_a, col_b = st.columns(2)

                with col_a:
                    st.success(f"Predicted topic: {pred_label}")
                    st.caption(
                        "This prediction is based on a classifier trained on cluster assignments derived from English scientific abstracts.")
                    st.write(f"Predicted cluster: {pred_cluster}")

                with col_b:
                    if confidence is not None:
                        st.metric("Confidence", round(confidence, 3))

                st.info(pred_explanation)

                if top_predictions is not None and not top_predictions.empty:
                    merged_preds = top_predictions.merge(
                        cluster_labels[["cluster", "cluster_label"]],
                        on="cluster",
                        how="left"
                    )
                    merged_preds["cluster_label"] = merged_preds["cluster_label"].fillna(
                        merged_preds["cluster"].apply(lambda x: f"Cluster {x}")
                    )
                    merged_preds["probability"] = merged_preds["probability"].round(3)

                    st.markdown("### Top predicted topics")
                    st.dataframe(
                        merged_preds[["cluster", "cluster_label", "probability"]],
                        use_container_width=True
                    )
            else:
                st.warning("Please paste some text before predicting.")
    else:
        st.info(
            "The live classifier is not available yet because the serialized model files were not found.\n\n"
            "To enable live inference, add these files to the project:\n"
            "- tfidf_vectorizer.joblib\n"
            "- cluster_classifier.joblib\n\n"
            "Once added, the app will allow users to paste a new abstract and receive a predicted topic in real time."
        )

    st.markdown("---")
    st.markdown("## 🔎 Cluster Explorer")

    st.sidebar.header("Filters")

    clusters = sorted(docs["cluster"].dropna().unique().tolist())

    cluster_option_map = {}
    for c in clusters:
        row = cluster_labels[cluster_labels["cluster"] == c]
        if not row.empty:
            cluster_option_map[f"{c} — {row['cluster_label'].iloc[0]}"] = c
        else:
            cluster_option_map[f"{c} — Cluster {c}"] = c

    journal_options = ["All"]
    if "journal" in docs.columns:
        journal_counts_all = docs["journal"].fillna("Unknown").value_counts()
        valid_journals = journal_counts_all[journal_counts_all > 2].index.tolist()
        journal_options = ["All"] + sorted(valid_journals)

    with st.sidebar.form("filtros_form"):
        selected_cluster_label = st.selectbox(
            "Select cluster",
            list(cluster_option_map.keys())
        )
        selected_cluster = cluster_option_map[selected_cluster_label]

        st.info("""
        **Cluster distribution note**
        
        The clustering results show an imbalanced distribution of records across clusters:
        
        - **Cluster 0:** 860 records
        - **Cluster 1:** 130 records
        - **Cluster 2:** 2 records
        - **Cluster 3:** 7 records
        - **Cluster 4:** 1 record
        
        Some clusters contain a very small number of records. In future iterations, it will be important to review this distribution carefully, evaluate whether these small clusters represent meaningful niche topics or clustering artifacts, and improve the clustering strategy over time.
        """)

        n_examples = st.slider("Number of representative documents", 3, 15, 5)
        search_term = st.text_input("Search keyword in abstracts")
        selected_journal = st.selectbox("Filter by journal", journal_options)
        st.form_submit_button("Apply filters")

    cluster_docs = docs[docs["cluster"] == selected_cluster].copy()

    if search_term:
        cluster_docs = cluster_docs[
            cluster_docs["abstract_clean"].fillna("").str.contains(search_term, case=False, na=False)
        ]

    if selected_journal != "All" and "journal" in cluster_docs.columns:
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
        st.metric("Documents", len(cluster_docs))

    with col2:
        if not cluster_stats.empty and "avg_abstract_length" in cluster_stats.columns:
            st.metric(
                "Avg. abstract length",
                round(float(cluster_stats["avg_abstract_length"].iloc[0]), 2)
            )
        else:
            st.metric("Avg. abstract length", "N/A")

    with col3:
        if not cluster_label_row.empty:
            st.metric("Estimated topic", cluster_label_row["cluster_label"].iloc[0])
        else:
            st.metric("Estimated topic", "N/A")

    if not cluster_words.empty:
        st.caption("Top signal words: " + ", ".join(cluster_words["word"].head(5).tolist()))

    tab1, tab2, tab3 = st.tabs([
        "Top Words",
        "Representative Documents",
        "Journal Distribution"
    ])

    with tab1:
        st.markdown("### Most frequent words")
        if cluster_words.empty:
            st.info("No words available for this cluster.")
        else:
            top_n_words = st.slider("Number of words to display", 5, 20, 10)

            plot_df = cluster_words.head(top_n_words).sort_values("count", ascending=True)

            fig = px.bar(
                plot_df,
                x="count",
                y="word",
                orientation="h",
                title=f"Top words for cluster {selected_cluster}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cluster_words.head(top_n_words), use_container_width=True)

    with tab2:
        st.markdown("### Most representative documents")

        sample_docs = representative_docs[
            representative_docs["cluster"] == selected_cluster
        ].copy()

        if search_term:
            sample_docs = sample_docs[
                sample_docs["abstract_clean"].fillna("").str.contains(search_term, case=False, na=False)
            ]

        if selected_journal != "All" and "journal" in sample_docs.columns:
            sample_docs = sample_docs[sample_docs["journal"] == selected_journal]

        sample_docs = sample_docs.head(n_examples)

        if sample_docs.empty:
            st.info("No documents available for the selected filters.")
        else:
            for _, row in sample_docs.iterrows():
                title = row["title"] if pd.notna(row["title"]) else "Untitled"
                st.markdown(f"**{title}**")

                if "journal" in row and pd.notna(row["journal"]):
                    st.caption(f"Journal: {row['journal']}")

                abstract = str(row["abstract_clean"]) if pd.notna(row["abstract_clean"]) else ""
                short_abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                st.write(short_abstract)

                if "representative_score" in row:
                    st.caption(f"Representative score: {row['representative_score']}")

                with st.expander("View full abstract"):
                    st.write(abstract)

                st.markdown("---")

    with tab3:
        st.markdown("### Most frequent journals")
        if "journal" in cluster_docs.columns:
            journal_counts = (
                cluster_docs["journal"]
                .fillna("Unknown")
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
                title=f"Top journals for cluster {selected_cluster}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(journal_counts, use_container_width=True)
        else:
            st.info("No journal column available in exported data.")

    st.markdown("---")
    st.markdown(
        "This app uses notebook-exported results: "
        "`clustered_docs.csv`, `cluster_stats.csv`, `tfidf_vectorizer.joblib`, and `cluster_classifier.joblib`."
    )


if __name__ == "__main__":
    main()
