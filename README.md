# Scientific Clusters Explorer (NLP + Streamlit)

**Live App:** https://app-jsj8yakkavjbmwk9q539ye.streamlit.app/

---

## Overview

This project is an **interactive NLP application** built with Streamlit that allows users to explore thematic clusters of scientific articles based on their abstracts.

It combines **text mining, clustering, and interactive visualization** to transform unstructured scientific data into an intuitive exploration tool.

---

## Objective

The goal of this project is to demonstrate how to:

- Process and analyze large-scale scientific text data
- Apply NLP techniques to extract meaningful patterns
- Build an interactive app for exploratory analysis
- Bridge data science and user-facing applications

---

## Methodology

The pipeline follows these steps:

1. **Text preprocessing**
   - Cleaning and normalization of abstracts

2. **Feature extraction**
   - TF-IDF vectorization

3. **Clustering**
   - KMeans to identify thematic groups

4. **Post-analysis**
   - Top words per cluster
   - Representative documents
   - Cluster-level statistics

---

## App Features

The Streamlit app allows users to:

- Select and explore clusters
- View most frequent terms per cluster
- Inspect representative documents (titles + abstracts)
- Filter results by keyword and journal
- Analyze journal distribution
- Explore temporal trends (when available)

---

## Data Quality Handling

During development, inconsistencies were detected in the `journal` field, where some entries contained fragments of abstracts instead of actual journal names.

To address this:
- Filtering rules were applied (length, token count, frequency)
- Only valid journal values are displayed in the app

This highlights real-world data challenges and practical cleaning strategies.

---

## Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Plotly**

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/Streamlit.git
cd Streamlit
pip install -r requirements.txt
streamlit run streamlit_app_fixed.py
