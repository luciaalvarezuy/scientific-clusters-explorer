# Scientific Clusters Explorer: NLP, Clustering, and Interactive Topic Interpretation

**Live App:** https://app-jsj8yakkavjbmwk9q539ye.streamlit.app/

---

## Overview
This project is an interactive NLP application built with Streamlit that allows users to:

- explore thematic clusters of scientific articles  
- understand topics through automatic interpretation  
- classify new text in real time  

It combines text preprocessing, TF-IDF vectorization, unsupervised clustering, and a lightweight supervised model to transform unstructured scientific text into an interpretable and interactive exploration tool.

The app not only shows clusters, but also generates:
- automatic topic labels  
- short cluster explanations  
- representative documents for each cluster  
- real-time predictions for new input text  

---

## Objective
The goal of this project is to demonstrate how to:

- process and analyze scientific text data  
- apply NLP techniques to discover thematic patterns  
- interpret unsupervised clusters in a user-friendly way  
- bridge unsupervised and supervised learning  
- build an interactive app that connects data science with real user interaction  

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

### 6. Live text classification
- training a supervised classifier using cluster assignments as pseudo-labels  
- transforming new input text using the trained TF-IDF vectorizer  
- predicting the most likely topic/cluster in real time  
- returning confidence scores and top predictions  

### 7. Interactive exploration
- Streamlit dashboard with filters, term exploration, journal analysis, and live inference  

---

## App Features
The Streamlit app allows users to:

### Cluster exploration
- select and explore clusters  
- view automatic topic labels  
- read short explanations of each cluster  
- inspect the most frequent terms  
- view representative documents ranked by relevance  
- filter documents by keyword and journal  
- analyze journal distribution  

### Live text classifier
- paste a scientific abstract or short text  
- receive a predicted topic and cluster  
- view confidence score (when available)  
- inspect top predicted topics  

---

## Live Text Classifier: How it works
The classifier is built on top of the clustering pipeline:

1. TF-IDF vectorization is learned from the original dataset  
2. KMeans generates cluster assignments (unsupervised)  
3. These clusters are used as **pseudo-labels**  
4. A supervised model (Logistic Regression) is trained  
5. New text is transformed and classified in real time  

This approach connects:
- unsupervised learning → structure discovery  
- supervised learning → real-time inference  

---

## Language Considerations
The model was trained primarily on **English scientific abstracts**.

This means:
- best performance is expected for English input  
- the app may still accept other languages  
- however, predictions for Spanish or other languages may be less reliable  

This reflects a realistic NLP limitation when models are trained on domain-specific corpora.

---

## Why this project matters
This project goes beyond a typical NLP notebook and demonstrates the full pipeline from data to product.

It highlights skills in:
- NLP and text preprocessing  
- unsupervised learning (KMeans)  
- feature engineering (TF-IDF)  
- model interpretation  
- supervised modeling for inference  
- building interactive data products (Streamlit)  

---

## What makes this app different
Unlike a standard clustering demo, this application adds:

- an interpretation layer (labels + explanations)  
- representative document selection  
- a real-time classification system  

This transforms the project from:
- static analysis  
into  
- an interactive NLP application with inference capabilities  

---

## Data Quality Handling
During development, inconsistencies were detected in the `journal` field, where some entries contained fragments of abstracts instead of actual journal names.

To address this:
- filtering rules were applied based on frequency and plausibility  
- only valid journal values are displayed  

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
- Logistic Regression (classification)  

---

## Run Locally
```bash
git clone https://github.com/luciaalvarezuy/scientific-clusters-explorer.git
cd scientific-clusters-explorer
pip install -r requirements.txt
streamlit run streamlit_app_fixed.py
