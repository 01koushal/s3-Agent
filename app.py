import os
import duckdb
from groq import Groq
from flask import Flask, render_template, request, jsonify

# ---------------- CONFIG ----------------

# 🔐 AWS CREDENTIALS (FROM ENV)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")


# 🤖 GROQ MODEL
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# --------------------------------------

app = Flask(__name__)

# -------- Initialize Groq --------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def format_readable(result):
    # Single value (count, avg)
    if len(result.columns) == 1 and len(result) == 1:
        return str(result.iloc[0, 0])

    # Single record
    if len(result) == 1:
        row = result.iloc[0]
        lines = [f"{col.replace('_',' ').title()}: {row[col]}" for col in result.columns]
        return "\n".join(lines)

    # Large result set
    if len(result) > 10:
        return (
            f"Found {len(result)} matching records.\n"
            f"Example patient IDs: "
            + ", ".join(result["patient_id"].head(5).tolist())
        )

    # Small list
    return (
        f"Found {len(result)} records:\n"
        + ", ".join(result["patient_id"].tolist())
    )

# -------- Load data ONCE (startup) --------
def load_data():
    print("📥 Loading Parquet data from S3...")

    con = duckdb.connect(database=":memory:")

    # ✅ REQUIRED FOR S3
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # ✅ EXPLICIT AWS CONFIG FOR DUCKDB
    con.execute(f"SET s3_region='{AWS_REGION}';")
    con.execute(f"SET s3_access_key_id='{AWS_ACCESS_KEY_ID}';")
    con.execute(f"SET s3_secret_access_key='{AWS_SECRET_ACCESS_KEY}';")
    con.execute("SET s3_url_style='path';")

    tables = {
        "patients": "s3://my-healthcare-analyticsdata/data_parquet/patients/*.parquet",
        "prescriptions": "s3://my-healthcare-analyticsdata/data_parquet/prescriptions/*.parquet",
        "visits": "s3://my-healthcare-analyticsdata/data_parquet/visits/*.parquet",
    }

    for table, path in tables.items():
        print(f"➡️ Loading {table}")
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT * FROM read_parquet('{path}');
        """)

    print("✅ All Parquet data loaded successfully")
    return con

# Load at startup
con = load_data()

# -------- Collect schema --------
def get_schema(con):
    schema = {}
    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()

    for table in tables:
        cols = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
        schema[table] = cols["name"].tolist()

    return schema

SCHEMA = get_schema(con)

# -------- Intent Classification --------
def classify_intent(question):
    prompt = f"""
Classify the user question into ONE category:
- analytics : questions about healthcare data
- chat : greetings or help
- invalid : unrelated or unsafe questions

Return ONLY ONE WORD.

Question:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()

# -------- Question → SQL --------
def question_to_sql(question):
    schema_text = "\n".join(
        [f"- {table}({', '.join(cols)})" for table, cols in SCHEMA.items()]
    )

    prompt = f"""
You are an expert healthcare data analyst.

Available tables:
{schema_text}

IMPORTANT RULES:
- Medical conditions (like asthma, diabetes, copd, hypertension)
  are stored in the COLUMN patients.chronic_conditions
- chronic_conditions is a comma-separated STRING
- To check a condition, use:
  patients.chronic_conditions ILIKE '%condition%'
- DO NOT assume there is a separate table for conditions
- ALWAYS use the patients table for age, gender, and conditions
- Output ONLY a valid SQL query
- SQL must start with SELECT
- Do NOT explain anything
- Do NOT use DROP, DELETE, UPDATE, INSERT, ALTER

Question:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Convert questions into SQL queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# -------- SQL Safety --------
def is_safe_sql(sql):
    sql = sql.lower()
    forbidden = ["drop", "delete", "update", "insert", "alter"]
    return sql.startswith("select") and not any(w in sql for w in forbidden)

# -------- API --------
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()

    if not question:
        return jsonify({"answer": "❌ Question is required"}), 400

    intent = classify_intent(question)

    if intent == "chat":
        return jsonify({
            "answer": "🙂 I am a healthcare analytics assistant. Ask me about patients, prescriptions, and visits."
        })

    if intent == "invalid":
        return jsonify({
            "answer": "❌ I can only answer healthcare analytics related questions."
        })

    sql = question_to_sql(question)

    if not is_safe_sql(sql):
        return jsonify({"answer": "❌ Unsafe query blocked"}), 400

    try:
        result = con.execute(sql).fetchdf()
        answer = format_readable(result)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"}), 500

# -------- UI --------
@app.route("/")
def index():
    return render_template("index.html")

# -------- Entry Point --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
