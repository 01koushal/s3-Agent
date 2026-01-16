# S3 Agent

S3 Agent is a lightweight Flask application that lets you query **Parquet files stored in AWS S3** using plain English.
Instead of manually writing SQL, you can ask questions in natural language and get results instantly.

The app uses **DuckDB** to query Parquet files directly from S3 and an **LLM (Groq)** to translate questions into SQL.

---

## What this project does

* Reads Parquet files directly from AWS S3
* Converts natural language questions into SQL
* Runs queries using DuckDB (no database setup required)
* Returns results through a simple Flask API / UI

This is mainly built as an **experiment / learning project** for working with:

* S3 data
* DuckDB
* LLM-based query generation

---

## Tech Used

* Python
* Flask
* DuckDB
* AWS S3
* Groq LLM API

---

## Project Structure

```
s3-Agent/
│
├── app.py            # Flask app and query logic
├── requirements.txt  # Dependencies
├── templates/
│   └── index.html    # Basic UI
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region

S3_PARQUET_PATH=s3://bucket-name/path/*.parquet
GROQ_API_KEY=your_groq_api_key
```

> Credentials are expected to be provided via environment variables.

---

## Run the app

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

---

## Example Queries

You can ask questions like:

* How many patients have diabetes?
* How many male patients are there?
* Show patients from Delhi
* List patients with alcohol drinking habit

The app will automatically generate SQL and return the results.

---

## How it works (simple explanation)

1. User enters a question
2. The LLM converts it into a SQL query
3. DuckDB runs the query directly on S3 Parquet files
4. Results are sent back to the UI / API

No data is copied locally.

---

## Notes

* This project is intended for learning and experimentation
* Error handling and security can be improved
* Not production-ready in its current form

---

## Possible Improvements

* Better prompt control for SQL generation
* Authentication and access control
* Query history and logging
* UI improvements


