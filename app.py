import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from nlp_engine import IntentEngine
from analytics_engine import AnalyticsEngine
from rag_engine import RAGEngine

load_dotenv()
app = Flask(__name__)

intent_engine = IntentEngine()
analytics_engine = AnalyticsEngine()
rag_engine = RAGEngine()

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "analytics_ready": analytics_engine.ready,
        "rag_ready": rag_engine.ready,
        "intent_model_ready": intent_engine.ready
    })

@app.get("/api/metrics")
def metrics():
    return jsonify(intent_engine.evaluate())

@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 1500:
        return jsonify({"error": "Please keep the question under 1,500 characters."}), 400

    intent_result = intent_engine.predict(question)
    intent = intent_result["intent"]

    try:
        if intent == "analytics":
            result = analytics_engine.ask(question)
        elif intent == "knowledge":
            result = rag_engine.ask(question)
        elif intent == "chat":
            lower = question.lower()
            if any(term in lower for term in ["who are you", "what is your name", "introduce yourself", "tell me about yourself"]):
                answer = "I'm S3 Agent, a healthcare information and analytics assistant. I use NLP to identify your question's intent, retrieve explanations from the healthcare knowledge base with RAG, and query structured hospital datasets with Text-to-SQL. I provide educational information and data analysis, not medical diagnosis."
            elif any(term in lower for term in ["what can you do", "how can you help", "capabilities"]):
                answer = "I can explain diseases, symptoms, hospital services and general medicine information using the healthcare reference library. I can also answer numerical questions about the available patient, visit and prescription datasets. Try asking 'What is hypertension?' or 'How many patients have diabetes?'"
            elif any(term in lower for term in ["thank", "thanks"]):
                answer = "You're welcome! Ask me a healthcare question or a question about the hospital datasets whenever you're ready."
            elif any(term in lower for term in ["bye", "goodbye"]):
                answer = "Goodbye! Take care, and feel free to return whenever you have a healthcare knowledge or dataset question."
            else:
                answer = "Hello! I'm S3 Agent. I can explain healthcare topics using the reference library and analyse structured hospital data. What would you like to know?"
            result = {"answer": answer, "mode": "conversation", "sources": [], "sql": None}
        else:
            result = {
                "answer": "I can help with healthcare dataset analytics or questions grounded in the available medical reference documents. Please try one of those topics.",
                "mode": "out_of_scope",
                "sources": [],
                "sql": None
            }
        result["intent"] = intent
        result["intent_confidence"] = round(intent_result["confidence"], 4)
        result["intent_method"] = intent_result["method"]
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Question processing failed")
        return jsonify({
            "error": str(exc),
            "intent": intent,
            "intent_confidence": round(intent_result["confidence"], 4)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG") == "1")
