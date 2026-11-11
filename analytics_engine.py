import os
import re
from pathlib import Path

import duckdb
import pandas as pd
from groq import Groq

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
TABLE_FILES = {
    "patients": DATA / "patients.csv",
    "prescriptions": DATA / "prescriptions.csv",
    "visits": DATA / "visits.csv"
}

class AnalyticsEngine:
    def __init__(self):
        self.connection = duckdb.connect(database=":memory:")
        self.ready = False
        self._load_tables()

    def _load_tables(self):
        loaded = []
        for table, path in TABLE_FILES.items():
            if path.exists():
                safe_path = str(path).replace("'", "''")
                self.connection.execute(
                    f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{safe_path}', header=true)"
                )
                loaded.append(table)
        # Keep the app runnable even if the original CSVs have not yet been copied.
        if not loaded:
            self.connection.execute("""
                CREATE TABLE patients(patient_id VARCHAR, age INTEGER, gender VARCHAR,
                    chronic_conditions VARCHAR, bmi DOUBLE, smoker_status BOOLEAN, alcohol_use BOOLEAN);
                INSERT INTO patients VALUES
                ('DEMO001',45,'Female','Diabetes',29.4,false,false),
                ('DEMO002',52,'Male','Hypertension',31.2,true,false),
                ('DEMO003',34,'Female','None',23.1,false,false),
                ('DEMO004',61,'Male','Diabetes; Hypertension',28.8,false,true);
                CREATE TABLE prescriptions(prescription_id VARCHAR, patient_id VARCHAR, medication VARCHAR);
                INSERT INTO prescriptions VALUES
                ('RX001','DEMO001','Metformin'),('RX002','DEMO002','Amlodipine'),
                ('RX003','DEMO004','Metformin');
                CREATE TABLE visits(visit_id VARCHAR, patient_id VARCHAR, visit_date DATE, visit_type VARCHAR);
                INSERT INTO visits VALUES
                ('V001','DEMO001','2025-01-10','Routine'),('V002','DEMO002','2025-02-12','Follow-up'),
                ('V003','DEMO004','2025-03-05','Routine');
            """)
        self.ready = True

    def schema_text(self):
        tables = self.connection.execute("SHOW TABLES").fetchdf()["name"].tolist()
        lines = []
        for table in tables:
            cols = self.connection.execute(f"PRAGMA table_info('{table}')").fetchdf()["name"].tolist()
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)

    def _generate_sql(self, question):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY in your .env file to enable natural-language SQL generation.")
        client = Groq(api_key=key)
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        prompt = f"""
Convert the user's healthcare dataset question into ONE DuckDB SELECT query.
Available schema:
{self.schema_text()}

Rules:
- Output SQL only, no markdown.
- Only SELECT statements are allowed. No CTE that modifies data.
- Use only the listed tables and columns.
- Do not invent columns.
- For chronic_conditions text matching, use lower(CAST(chronic_conditions AS VARCHAR)) LIKE '%term%'.
- Never provide medical advice; this is dataset analytics.
Question: {question}
"""
        response = client.chat.completions.create(
            model=model, messages=[
                {"role": "system", "content": "You generate safe, read-only DuckDB SQL."},
                {"role": "user", "content": prompt}
            ], temperature=0
        )
        return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

    def _validate_sql(self, sql):
        normalized = sql.strip()
        if not re.match(r"(?is)^select\b", normalized):
            raise ValueError("Generated query was blocked: only SELECT statements are allowed.")
        if ";" in normalized.rstrip(";"):
            raise ValueError("Generated query was blocked: multiple SQL statements are not allowed.")
        forbidden = r"\b(insert|update|delete|drop|alter|create|copy|attach|detach|install|load|pragma|export|import|call)\b"
        if re.search(forbidden, normalized, flags=re.I):
            raise ValueError("Generated query was blocked by the read-only SQL safety check.")
        return normalized.rstrip(";")

    def ask(self, question):
        sql = self._validate_sql(self._generate_sql(question))
        result = self.connection.execute(sql).fetchdf()
        preview = result.head(50).to_dict(orient="records")
        if len(result) == 0:
            answer = "The query ran successfully, but no matching records were found."
        elif len(result) == 1 and len(result.columns) == 1:
            answer = f"Result: {result.iloc[0, 0]}"
        else:
            answer = f"The query returned {len(result)} row(s). Here are the results."
        return {
            "answer": answer,
            "mode": "structured_analytics",
            "sql": sql,
            "columns": list(result.columns),
            "rows": preview,
            "row_count": int(len(result)),
            "sources": [],
            "dataset_mode": "original_csv" if any(p.exists() for p in TABLE_FILES.values()) else "synthetic_demo"
        }
