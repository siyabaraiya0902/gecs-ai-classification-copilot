# Morningstar GECS AI Classification Copilot

This project is an AI-powered classification Copilot for GECS-style industry and subindustry prediction.

## What It Does

- Accepts company long profile, segment name, and segment description
- Uses FastAPI as the model backend
- Uses Streamlit as the analyst-facing interface
- Connects Task 1 industry classification and Task 2 subindustry classification models
- Returns predicted industry, predicted subindustry, confidence score, routing decision, and top alternatives
- Uses a GECS taxonomy lookup file to convert codes into readable names
- Captures analyst feedback for future retraining

## Tech Stack

Python, FastAPI, Streamlit, scikit-learn, TF-IDF, Linear SVM, Logistic Regression, joblib, pandas

## How to Run Locally

Start the FastAPI backend:

```bash
python -m uvicorn main:app --reload
- Run API Integration.webm for the demo 
