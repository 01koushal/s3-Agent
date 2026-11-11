import csv
import os
import re
from pathlib import Path

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "intent_dataset.csv"

SEED_EXAMPLES = {
    "analytics": [
        "how many patients are in the dataset", "count patients with diabetes",
        "show female patients older than 40", "calculate average patient bmi",
        "what are the most common prescriptions", "show total number of visits",
        "group patients by gender", "find patients who smoke",
        "list patients with hypertension", "compare patient counts by age group",
        "which medicine was prescribed most often", "show visits by month",
        "find patients with bmi above 30", "count male patients",
        "give me the average age of patients", "how many prescriptions exist",
        "show patients from Hyderabad", "calculate the maximum bmi",
        "list the ten youngest patients", "filter visits for a patient"
    ],
    "knowledge": [
        "what are common risk factors for diabetes", "explain the symptoms of hypertension",
        "what is type two diabetes", "how can high blood pressure be managed",
        "explain the difference between type one and type two diabetes",
        "what are common complications of diabetes", "describe cardiovascular disease risk factors",
        "what does insulin do in the body", "explain cholesterol and heart disease",
        "what lifestyle factors affect blood pressure", "what is atherosclerosis",
        "summarize the medical document about diabetes", "according to the reference guide what is hypertension",
        "what are warning signs of a stroke", "explain the role of physical activity in heart health",
        "what does the knowledge base say about nutrition"
    ],
    "chat": [
        "hello", "hi there", "good morning", "hey how are you",
        "what can you do", "help me use this application", "thank you",
        "thanks a lot", "who are you", "good evening", "how does this system work"
    ],
    "invalid": [
        "write a poem about the moon", "what is the capital of France",
        "tell me a joke about football", "book a flight to Delhi",
        "who won the cricket match", "generate a gaming username",
        "what is the latest movie", "help me hack an account",
        "write a Java program for sorting", "what is the weather today",
        "recommend a gaming laptop", "tell me a celebrity's phone number"
    ]
}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = text.split()
    return " ".join(tokens)

class IntentEngine:
    def __init__(self):
        self.pipeline = None
        self.ready = False
        self.training_size = 0
        self._train()

    def _load_examples(self):
        examples = []
        if DATASET.exists():
            with DATASET.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("text") and row.get("intent"):
                        examples.append((row["text"], row["intent"].strip().lower()))
        if len(examples) < 30:
            examples = [(text, label) for label, texts in SEED_EXAMPLES.items() for text in texts]
        return examples

    def _train(self):
        examples = self._load_examples()
        X = [preprocess_text(x) for x, _ in examples]
        y = [label for _, label in examples]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        self.pipeline.fit(X, y)
        self.training_size = len(examples)
        self.ready = True

    def predict(self, text):
        cleaned = preprocess_text(text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        label = str(classes[best])

        # Explicit high-precision conversational rules keep greetings and identity
        # questions out of medical retrieval and analytics.
        chat_patterns = [
            r"^(hi|hello|hey|good morning|good afternoon|good evening)( there)?$",
            r"\b(who are you|what is your name|introduce yourself|tell me about yourself)\b",
            r"\b(what can you do|how can you help|how do i use|how does this (app|application|system) work)\b",
            r"^(thanks|thank you|bye|goodbye)( so much)?$"
        ]
        if any(re.search(pattern, cleaned) for pattern in chat_patterns):
            return {"intent": "chat", "confidence": max(confidence, 0.95), "method": "conversation_rule_override"}

        # Numeric/aggregate requests over hospital records should route to SQL.
        analytics_signal = re.search(
            r"\b(how many|count|number of|total|average|mean|median|maximum|minimum|highest|lowest|most common|top \d+|distribution|breakdown|compare|group by|per month|by month|by year|percentage|percent)\b",
            cleaned
        )
        dataset_signal = re.search(
            r"\b(patient|patients|visit|visits|prescription|prescriptions|admission|admissions|appointment|appointments|record|records|diagnosis|diagnoses|doctor|doctors|medicine|medicines)\b",
            cleaned
        )
        if analytics_signal and dataset_signal:
            return {"intent": "analytics", "confidence": max(confidence, 0.90), "method": "analytics_pattern_override"}

        # Low-confidence healthcare questions go to the knowledge route.
        if confidence < 0.38:
            medical = {"health", "patient", "diabetes", "blood", "pressure", "heart", "medical", "disease", "medicine", "hospital", "symptom", "treatment"}
            if any(token in cleaned.split() for token in medical):
                label = "knowledge"
            else:
                label = "invalid"
            method = "tfidf_logistic_regression_low_confidence_fallback"
        else:
            method = "tfidf_logistic_regression"
        return {"intent": label, "confidence": confidence, "method": method}

    def evaluate(self):
        examples = self._load_examples()
        X = [preprocess_text(x) for x, _ in examples]
        y = [label for _, label in examples]
        if len(set(y)) < 2:
            return {"training_examples": len(y), "message": "Not enough intent classes to evaluate."}
        # Report a reproducible holdout estimate when each class has enough examples.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return {
                "training_examples": len(X),
                "holdout_samples": len(y_test),
                "holdout_accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                "note": "Metrics are calculated on the included labelled intent examples; expand and manually validate this dataset for a meaningful academic evaluation."
            }
        except ValueError as exc:
            return {"training_examples": len(X), "message": str(exc)}
