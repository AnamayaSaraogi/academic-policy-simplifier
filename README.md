# Academic Policy Simplifier

Academic Policy Simplifier is an AI-powered Streamlit application that analyzes institutional policy documents (PDF or text) and transforms them into structured, searchable, and understandable insights.

Live Application:
https://academic-policy-simplifier-j352y54z6eeykrxzxzk5cd.streamlit.app/
---

## Overview

Institutional policies are often long, complex, and difficult to interpret. This application applies Natural Language Processing (NLP) techniques to:

- Extract and clean policy text
- Generate concise summaries
- Identify key academic rules
- Provide intelligent question answering
- Auto-generate suggested questions based on policy content
- Visualize the system processing pipeline

The goal is to convert unstructured academic regulations into an interactive AI assistant.

---

## Key Features

### 1. Policy Upload
- Supports PDF and TXT files
- Direct text input supported
- Automatic text extraction and cleaning

### 2. Intelligent Summary Generation
- Filters malformed or noisy sentences
- Scores sentences using word frequency
- Returns coherent, high-quality summaries

### 3. Rule Extraction
Automatically detects:
- Attendance requirements
- Condonation limits
- Penalties and fines
- Deadlines
- Eligibility criteria

### 4. AI Q&A Assistant
- Hybrid semantic + keyword matching
- Uses spaCy similarity scoring
- Returns ranked relevant sentences
- Supports contextual academic queries

### 5. Suggested Question Generator
- Dynamically generates context-aware questions
- Tailored to the uploaded policy
- Makes the system feel like an AI assistant

### 6. Transparent System Architecture View
- Shows each preprocessing stage:
  - Raw extraction
  - Cleaning
  - Sentence splitting
  - Token filtering
  - Rule extraction
- Includes visual pipeline diagram

---

## Technology Stack

Frontend:
- Streamlit

Backend / NLP:
- spaCy (en_core_web_sm)
- NLTK
- Scikit-learn
- PyPDF2

Data Processing:
- NumPy
- Counter
- Regular Expressions

Visualization:
- Matplotlib
- WordCloud
- NetworkX

---

## System Architecture

The processing pipeline follows these stages:

1. Document Upload
2. Text Extraction
3. Cleaning and Normalization
4. Sentence Segmentation
5. Tokenization and Stopword Removal
6. Rule Detection
7. Summary Generation
8. Semantic Question Answering

The architecture diagram is included within the application under the "Architecture" tab.

---

## How to Run Locally

Clone the repository:

git clone https://github.com/AnamayaSaraogi/academic-policy-simplifier.git

Navigate into the folder:

cd academic-policy-simplifier

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

## Deployment

The application is deployed on Streamlit Community Cloud.

Live URL:
https://academic-policy-simplifier-j352y54z6eeykrxzxzk5cd.streamlit.app/


- Conflict detection between rules
- Risk-level highlighting of penalties
- Multi-document comparison
- Conversational memory mode
- Vector database integration
- Transformer-based summarization
