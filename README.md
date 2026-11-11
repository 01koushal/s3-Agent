# S3 Agent — Integrated NLP + Text-to-SQL + RAG

This version combines the original S3 Agent healthcare analytics workflow with the retrieval architecture from Koushal-RAG-Agent.

## Capabilities

- NLTK-based text normalization/tokenization
- TF-IDF feature extraction and Logistic Regression intent classifier
- Four intent classes: analytics, knowledge, chat, invalid
- Large reproducible intent-data generator (`generate_intent_data.py`) with 2,000+ unique starter examples
- Visible NLP intent badge and confidence on every assistant response
- Chat-first interface with a fixed bottom message composer and newest messages at the bottom
- Natural-language-to-SQL using Groq
- Read-only SQL execution through DuckDB
- RAG using Gemini embeddings, FAISS vector search, and Groq answer generation
- Retrieved source filenames and similarity scores
- Intent evaluation endpoint with accuracy, precision, recall, and F1
- Unified Flask UI

## 1. Setup

Python 3.10–3.12 is recommended.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys:

```env
GROQ_API_KEY=...
GEMINI_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

Never commit `.env` or API keys.

## 2. Generate the NLP training dataset

The repository includes `generate_intent_data.py`. Run it to create a larger, reproducible CSV for the TF-IDF + Logistic Regression classifier:

```bash
python generate_intent_data.py
```

It creates `data/intent_dataset.csv` with four labels: `chat`, `knowledge`, `analytics`, and `invalid`. The generated set contains over 2,000 unique template-based examples. It covers greetings and assistant identity, healthcare and hospital topics, diseases and medicine education, and numerical questions about structured records.

**Important academic note:** this is synthetic template-based augmentation, not a collection of real user questions. Inspect and improve the examples, add manually written paraphrases, and create a separate manually labelled test CSV before reporting final accuracy/F1. A large generated dataset alone does not prove real-world generalisation.

## 3. Healthcare analytics data

The included `data/` folder contains a small synthetic demonstration dataset so the app can run immediately.

For your original dataset, replace the demo CSVs with the original files from your S3 Agent repository:
- `data/patients.csv`
- `data/prescriptions.csv`
- `data/visits.csv`

The application detects these files automatically and loads them into DuckDB. The demo data is synthetic and must not be presented as real patient data.

## 4. Knowledge base

Place plain-text healthcare reference documents (`.txt`) inside `knowledge_base/`.
The healthcare `.txt` files are the RAG knowledge base. Build and save their embeddings/index before running the app:

```bash
python build_embeddings.py
```

This reads every `.txt` file, chunks the content, creates Gemini embeddings, normalizes the vectors, and saves the FAISS index plus chunk metadata:
- `storage/medical.faiss`
- `storage/chunks.pkl`

This is the same general workflow as your Koushal-RAG-Agent: documents → chunks → Gemini embeddings → FAISS index → similarity retrieval. The app loads the saved index when it starts. If you add or edit healthcare `.txt` files, run `python build_embeddings.py` again to rebuild it. Do not commit API keys; the generated index files may be committed if you deliberately want the repository to include the prebuilt index.

Use reputable, permitted reference material. Do not upload identifiable patient records into the RAG knowledge base.

## 5. Run

After building the embeddings, start Flask:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## 6. Example questions

### Analytics
- How many patients have diabetes?
- Show female patients older than 40.
- Calculate the average BMI.
- What are the most common prescriptions?

### Knowledge retrieval
- What are common risk factors for diabetes?
- Explain the symptoms of hypertension.
- What are warning signs of stroke?

### Chat / out-of-scope
- Hi, who are you?
- What can you do?
- Explain the hospital outpatient department
- What precautions matter when using antibiotics?
- Tell me a joke (out of scope)

## 7. API

- `GET /api/health` — service and configuration status
- `GET /api/metrics` — intent classifier holdout evaluation
- `POST /api/ask` — `{"question": "..."}`

The response includes the predicted intent, confidence, classification method, selected processing mode, answer, and (when relevant) SQL, result rows, retrieved sources, and similarity scores. The chat interface keeps the composer fixed at the bottom and scrolls the newest messages into view.

## 8. Academic evaluation checklist

Do not report placeholder or invented results. Run the evaluation endpoint and test the system using a separate manually labelled test set.

Recommended experiments:
1. Compare TF-IDF + Logistic Regression against LLM intent classification on the same held-out examples.
2. Report accuracy, precision, recall, and macro-F1 for intent classification.
3. Prepare question-to-SQL test cases with expected SQL/results and report execution accuracy.
4. For RAG, manually label relevant source passages and measure Precision@K / Recall@K.
5. Compare answers with and without retrieved context, checking factual grounding and source attribution.
6. Record latency and API usage for each route.

## Notes and limitations

- The generated intent examples are template-based augmentation, not real user questions. The very high holdout score on these examples may reflect template similarity; do not use it as a claim of real-world accuracy. Create a separate manually labelled test set.
- RAG requires both API keys and internet access for Gemini embeddings and Groq generation.
- Analytics requires GROQ_API_KEY. Without original CSVs, a clearly synthetic demo dataset is loaded.
- SQL safety checks are defensive application checks, not a complete security boundary. Use a restricted read-only database and review generated SQL before production use.
- This is an educational analytics and document-retrieval system, not a medical diagnostic system.
