# ESG-Insight-A-Multi-Level-NLP-System-for-Corporate-Sustainability

> **Automating ESG report analysis with NLP and Transformers**

---

## Overview

**ESGInsight** is an end-to-end **Natural Language Processing (NLP)** system that transforms unstructured corporate ESG reports into structured, interpretable insights.  
It integrates classical linguistic processing with modern deep learning models such as **FinBERT** (for sentiment analysis) and **BART** (for abstractive summarization) to evaluate Environmental, Social, and Governance performance objectively.

This project bridges the gap between qualitative ESG disclosures and quantitative intelligence — empowering stakeholders to assess sustainability performance efficiently, transparently, and at scale.

---

## System Architecture

The ESGInsight pipeline consists of **12 modular layers**:

1. **PDF Upload** – Upload ESG reports in PDF format.  
2. **Text Extraction** – Extract textual content using `PyMuPDF` and `pdfplumber`.  
3. **Text Preprocessing** – Tokenization, lemmatization, stopword removal, and normalization.  
4. **Linguistic Analysis** – POS tagging, dependency parsing, and chunking via `spaCy`.  
5. **Semantic Analysis** – Word embeddings using Word2Vec, GloVe, and BERT.  
6. **Sentiment & Topic Modeling** – ESG-specific sentiment detection (FinBERT) and topic discovery (LDA).  
7. **Pragmatic Analysis** – Coreference and discourse resolution for contextual coherence.  
8. **Generative Summarization** – Abstractive summaries via BART/T5.  
9. **ESG Scoring** – Weighted sentiment aggregation into a 0–100 ESG Score.  
10. **Embedding & Similarity** – Cosine similarity analysis across companies.  
11. **Visualization Layer** – WordClouds, bar charts, radar plots, and gauge indicators.  
12. **Dashboard & Output Layer** – JSON output and interactive dashboards for interpretation.

---

## Features

- 📄 Automated PDF text extraction and cleaning  
- 🔠 Multi-layer NLP analysis (syntax, semantics, sentiment, discourse)  
- 💬 ESG sentiment scoring using **FinBERT**  
- 🧩 Topic modeling using **LDA (Latent Dirichlet Allocation)**  
- 🧾 Abstractive summarization using **BART**  
- 📊 Interactive visualizations with **Matplotlib**, **Plotly**, and **WordCloud**  
- 🧮 JSON output for analytical integration  
- 🌐 Company-to-company ESG comparison

---

## Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **NLP Frameworks** | spaCy, NLTK, Gensim |
| **Transformer Models** | FinBERT, BART (via HuggingFace Transformers) |
| **Text Processing** | PyMuPDF, pdfplumber |
| **Visualization** | Matplotlib, Plotly, WordCloud |
| **Modeling** | LDA, TF-IDF |
| **Environment** | Google Colab / Jupyter Notebook |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ESGInsight.git
cd ESGInsight
```

## Install dependencies:

To deploy this project run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Example ```requirements.txt```

```bash
scipy
gensim
nltk
spacy
sentence-transformers
transformers
matplotlib
seaborn
wordcloud
plotly
pymupdf
pdfplumber
tqdm
networkx
```

## Usage (Colab / Local)

1. Upload your ESG report PDFs.
2.Open and run the notebook: NLP_Project_Final1.ipynb
3. The notebook will automatically:
- Extract and preprocess report text
- Perform sentiment & topic modeling
- Generate abstractive summaries
- Compute ESG scores and visualizations
- Save results as structured JSON

Output path (default):

```bash
/content/esg_results/esg_results.json
```

## Visualizations

- ESG pillar sentiment distribution
- Word cloud of most frequent ESG terms
- Radar chart of ESG pillar comparison
- Gauge chart for overall ESG score

## Future Scope

- Add multilingual ESG report analysis
- Integrate Large Language Models (LLMs) for improved contextual reasoning
- Implement Explainable AI (XAI) for transparent ESG scoring
- Enable real-time ESG trend monitoring via APIs
- Deploy an interactive web dashboard for company comparisons

## Conclusion 

ESGInsight showcases how AI and NLP can automate sustainability analytics. By integrating linguistic, semantic, and transformer-based methods, it translates verbose ESG disclosures into concise, data-driven intelligence — making ESG assessment faster, consistent, and transparent.

## Authors: 

This project is a collaborated work of: - [@AmishiDesai04](https://www.github.com/AmishiDesai04) , [@chahelgupta](https://www.github.com/chahelgupta) , Chaitanya Ajgaonkar, [@Armaanshah32](https://www.github.com/Armaanshah32)

