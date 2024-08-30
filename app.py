import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import fitz
import re
import torch
import pandas as pd
from fpdf import FPDF
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./fine-tuned-model')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')

# Load summarization model and tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.uploaded_files = None
        self.single_file = None

    def fit(self, X, y=None):
        if 'uploaded_files' in X:
            self.uploaded_files = X['uploaded_files']
        elif 'single_file' in X:
            self.single_file = X['single_file']
        else:
            raise ValueError("Input X must contain either 'uploaded_files' or 'single_file'.")
        return self

    def transform(self, X=None):
        if self.single_file:
            return {self.single_file.name: self.extract_text_from_pdf(self.single_file)}
        elif self.uploaded_files:
            return self.extract_texts_from_files(self.uploaded_files)
        else:
            raise ValueError("Either 'uploaded_files' or 'single_file' must be provided.")
    
    def extract_texts_from_files(self, files):
        documents = {}
        for file in files:
            documents[file.name] = self.extract_text_from_pdf(file)
        return documents
    
    def extract_text_from_pdf(self, file):
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        extracted_text = ''
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            extracted_text += text + "\n\n"
        return extracted_text

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, document_dict):
        if not isinstance(document_dict, dict):
            raise ValueError("Input must be a dictionary where keys are filenames and values are texts.")
        return {key: self.clean_text(text) for key, text in document_dict.items()}
    
    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s*[\u2022\u25AA\u25AB]\s*', ' ', text)
        text = re.sub(r'\s*-\s*', ' - ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip()

def summarize_text(text):
    max_chunk_size = 1024
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summary_parts = []
    for chunk in text_chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            summary_parts.append(summary)
        except Exception as e:
            print(f"Error summarizing text: {e}")

    return " ".join(summary_parts)

def create_cv_summary(cv_text):
    try:
        summary = summarize_text(cv_text)

        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        if len(sentences) < 4:
            sentences += [''] * (4 - len(sentences))
        
        para1 = ' '.join(sentences[:2])
        para2 = ' '.join(sentences[2:4])

        return f"{para1}\n\n{para2}"

    except Exception as e:
        print(f"Error processing CV: {e}")
        return "Error processing CV."

def summarize_cv_dict(cv_dict):
    summaries = {}
    for name, text in cv_dict.items():
        summaries[name] = create_cv_summary(text)
    return summaries

pipeline = Pipeline([
    ('text_extractor', TextExtractor()),
    ('text_cleaner', TextCleaner())
])

# Title of the app
st.title("CV and Job Description Processor")

# Section 1: Upload CVs
st.header("Upload CVs")
st.write("Please upload the CVs as PDF files.")
uploaded_cvs = st.file_uploader("Choose CV PDFs", type="pdf", accept_multiple_files=True)

# Process uploaded CVs if any
if uploaded_cvs:
    processed_cvs = pipeline.fit_transform({'uploaded_files': uploaded_cvs})
else:
    st.write("No CVs uploaded yet.")

st.write("---")

# Section 2: Job Description
st.header("Job Description")
job_desc_pdf = st.file_uploader("Upload Job Description PDF", type="pdf")

if job_desc_pdf:
    processed_job = list(pipeline.fit_transform({'single_file': job_desc_pdf}).values())

# Ranking by fine-tuned model
st.title("CV Ranking Based on Job Description")
rank_bt = st.button('Rank CVs')

if rank_bt:
    st.write("Processing CVs...")

    # Fine-tuned model ranking
    input_texts = [f"{cv} [SEP] {processed_job[0]}" for cv in processed_cvs.values()]
    tokenized_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        predictions = torch.softmax(outputs.logits, dim=1)[:, 1]

    ranked_cvs_model = sorted(zip(processed_cvs.keys(), predictions.numpy()), key=lambda x: x[1], reverse=True)
    ranked_df_model = pd.DataFrame(ranked_cvs_model, columns=["CV Filename", "Relevance Score"]).drop(columns=["Relevance Score"])
    ranked_df_model.index += 1
    
    st.write("Ranking by the fine-tuned model:")
    st.table(ranked_df_model)

    # Cosine similarity ranking
    def clean_text(text):
        tokens = text.lower().split()
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords]
        return ' '.join(tokens)

    cleaned_cvs = {name: clean_text(text) for name, text in processed_cvs.items()}
    cleaned_job = clean_text(processed_job[0])

    texts = list(cleaned_cvs.values()) + [cleaned_job]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    cv_tfidf_matrix = tfidf_matrix[:-1]
    job_tfidf_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(cv_tfidf_matrix, job_tfidf_vector.reshape(1, -1))

    relevance_scores = {cv_name: similarity_scores[i][0] for i, cv_name in enumerate(cleaned_cvs.keys())}
    sorted_cvs_cosine = sorted(relevance_scores.items(), key=lambda item: item[1], reverse=True)
    ranked_df_cosine = pd.DataFrame(sorted_cvs_cosine, columns=["CV Filename", "Relevance Score"]).drop(columns=["Relevance Score"])
    ranked_df_cosine.index += 1 

    st.write("Ranking by cosine similarity measure:")
    st.table(ranked_df_cosine)

summarize = st.button('Summarize CVs')
if summarize:
    summaries = summarize_cv_dict(processed_cvs)
    # Display summaries for each CV
    st.header("CV Summaries")
    for name, summary in summaries.items():
        st.subheader(f"Summary for {name}:")
        st.write(summary)
