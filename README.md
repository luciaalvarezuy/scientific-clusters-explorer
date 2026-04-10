# Scientific Clusters Explorer: NLP, Clustering, and Interactive Topic Interpretation

**Live App:** https://app-jsj8yakkavjbmwk9q539ye.streamlit.app/

---


## Overview
This project is an interactive NLP application built with Streamlit that allows users to explore thematic clusters of scientific articles based on their abstracts.

It combines text preprocessing, TF-IDF vectorization, unsupervised clustering, and interactive visualization to transform unstructured scientific text into an interpretable exploration tool.

The app not only shows clusters, but also generates:
- automatic topic labels
- short cluster explanations
- representative documents for each cluster

---

## Objective
The goal of this project is to demonstrate how to:

- process and analyze scientific text data
- apply NLP techniques to discover thematic patterns
- interpret unsupervised clusters in a user-friendly way
- build an interactive app that connects data science with real user exploration

---

## Methodology
The pipeline follows these steps:

### 1. Text preprocessing
- cleaning and normalization of abstracts

### 2. Feature extraction
- TF-IDF vectorization

### 3. Clustering
- KMeans to identify thematic groups of documents

### 4. Cluster interpretation
- extraction of top words per cluster
- automatic topic naming based on dominant lexical signals
- short automatic cluster descriptions

### 5. Representative documents
- ranking documents by overlap with top cluster terms
- surfacing the most representative abstracts per cluster

### 6. Interactive exploration
- Streamlit dashboard with filters, term exploration, and journal analysis

---

## App Features
The Streamlit app allows users to:

- select and explore clusters
- view automatic topic labels for each cluster
- read a short explanation of what each cluster represents
- inspect the most frequent terms per cluster
- view representative documents ranked by relevance
- filter documents by keyword and journal
- analyze journal distribution within each cluster

---
## Why this project matters
This project demonstrates the ability to move beyond notebook-based analysis and turn NLP outputs into an interactive product.

It highlights skills in:
- unsupervised learning
- text preprocessing
- cluster interpretation
- data storytelling
- Streamlit app development

---

## What makes this app different
Unlike a standard clustering demo, this application adds an interpretation layer on top of unsupervised learning.

Instead of showing only cluster IDs, the app provides:
- automatic cluster names
- short natural-language explanations
- representative documents selected using lexical overlap with top cluster words

This makes the results easier to understand for non-technical users and better suited for real exploratory workflows.

---

## Data Quality Handling
During development, inconsistencies were detected in the `journal` field, where some entries contained fragments of abstracts instead of actual journal names.

To address this:
- filtering rules were applied based on frequency and plausibility
- only valid journal values are displayed in the app

This reflects a realistic data-cleaning scenario and shows how exploratory tools must account for imperfect source data.

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- NLP preprocessing
- TF-IDF
- KMeans clustering

---

## Run Locally
```bash
git clone https://github.com/luciaalvarezuy/scientific-clusters-explorer.git
cd scientific-clusters-explorer
pip install -r requirements.txt
streamlit run streamlit_app_fixed.py
