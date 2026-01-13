import os
import duckdb
from groq import Groq
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------- CONFIG ----------------

# 🔐 AWS CREDENTIALS (TEMPORARY – LOCAL ONLY)
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "eu-north-1"   # Europe (Stockholm)

# 📦 S3 PARQUET LOCATION (FOLDER)
S3_PARQUET_PATH = (
    "s3://my-healthcare-analyticsdata/"
    "data_parquet/patients/*.parquet"
)

# 🤖 GROQ MODEL
GROQ_MODEL = "llama-3.1-8b-instant"

# --------------------------------------


# -------- Flask App --------
app = Flask(__name__)


# -------- Initialize Groq --------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# -------- Load data ONCE (important) --------
def load_data():
    print("📥 Loading Parquet data from S3...")

    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        CREATE TABLE patients AS
        SELECT * FROM read_parquet('{S3_PARQUET_PATH}');
    """)

    print("✅ Parquet data loaded successfully from S3")
    return con


# Load at startup
con = load_data()

COLUMNS = con.execute(
    "PRAGMA table_info('patients')"
).fetchdf()["name"].tolist()


# -------- Question → SQL (Groq) --------
def question_to_sql(question):
    prompt = f"""
You are an expert data analyst.

Table name: patients
Columns: {', '.join(COLUMNS)}

Rules:
- Output ONLY a valid SQL query
- SQL must start with SELECT
- Do NOT explain anything
- Do NOT use DROP, DELETE, UPDATE, INSERT, ALTER
- Assume all string values are lowercase

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


# -------- API: Ask question --------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "❌ Question is required"}), 400

    sql = question_to_sql(question)

    if not is_safe_sql(sql):
        return jsonify({"answer": "❌ Unsafe query blocked"}), 400

    try:
        result = con.execute(sql).fetchdf()

        if len(result.columns) == 1 and len(result) == 1:
            answer = str(result.iloc[0, 0])
        else:
            answer = result.to_dict(orient="records")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"}), 500


# -------- Serve UI --------
@app.route("/")
def index():
    return render_template("index.html")



# -------- Run --------
if __name__ == "__main__":
    app.run(debug=True)
