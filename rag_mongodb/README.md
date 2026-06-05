# rag_mongodb
# MongoDB RAG Agent

Simple RAG (Retrieval-Augmented Generation) pipeline built using n8n, MongoDB Atlas, and LLMs.

## Overview

This project lets you query a dataset using natural language.  
Instead of the model guessing, it retrieves relevant data and answers based on that.

Built mainly to understand how RAG systems work in practice.

## Flow

1. User sends a query
2. Relevant data is retrieved using MongoDB vector search
3. LLM generates response using retrieved context
4. Clean answer is returned

## Tech

- n8n
- MongoDB Atlas (vector search)
- Google Gemini / Groq
- Python (embedding + data ingestion)

## Embeddings

Data is processed using a Python script and stored in MongoDB with embeddings.  
These embeddings are used for semantic search during retrieval.

## Use Case

Currently set up with a healthcare-style dataset, but can be used for any domain by changing the data.

## Setup

- Import `workflow.json` into n8n
- Add required API keys
- Run the workflow

## Notes

- API keys are not included
- This is a learning project, not production-ready

## Author

Koushal
