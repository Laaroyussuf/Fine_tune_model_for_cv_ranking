# CV Relevance Ranking and Summary Generation
## Objective
This project aims to evaluate the ability to fine-tune a language model for specific NLP tasks, assess its performance, and provide a summary of results.

## Task Description
This repository contains a back-end service that processes a small dataset of CVs and a job description. The service performs two main tasks:

## Fine-Tune a Language Model

- Phase 1: Rank the CVs
Fine-tunes a BERT model to rank the CVs based on their relevance to a provided job description.
Uses cosine similarity to measure the relevance of each CV to the job description.
The goal is to prioritize the CVs that best match the job requirements.
- Phase 2: Generate Summaries
Generates a concise, two-paragraph summary for each CV.
Highlights the key skills and experiences relevant to the job description.
Implement a Back-End Service

Develops a script or service that takes a list of CVs and a job description as inputs.
Outputs:
The CVs ranked by their relevance to the job description using both the fine-tuned BERT model and cosine similarity.
A two-paragraph summary for each CV.
