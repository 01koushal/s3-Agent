"""Generate a large, reproducible starter intent dataset for S3 Agent.

This is template-based data augmentation, not real user data. Review examples,
remove duplicates, and keep a separately hand-labelled test set for evaluation.
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "intent_dataset.csv"
RNG = random.Random(42)

# Variations deliberately cover paraphrases, word order, and common phrasing.
ANALYTICS = {
    "entities": ["patients", "visits", "prescriptions", "hospital admissions", "medical records", "doctors", "appointments", "diagnoses"],
    "conditions": ["with diabetes", "with hypertension", "with heart disease", "older than 50", "from Hyderabad", "with a BMI above 30", "who visited this month", "with a prescription for metformin"],
    "metrics": ["total number", "count", "average", "minimum", "maximum", "percentage", "distribution", "monthly total"],
    "timeframes": ["this month", "this year", "last month", "by month", "by year", "between January and June", "for each quarter"],
    "groups": ["gender", "age group", "city", "diagnosis", "department", "month", "medicine", "doctor"],
    "verbs": ["show", "find", "list", "count", "calculate", "compare", "summarize", "group"],
}
KNOWLEDGE = {
    "diseases": ["diabetes", "type 1 diabetes", "type 2 diabetes", "hypertension", "high blood pressure", "asthma", "pneumonia", "heart disease", "coronary artery disease", "stroke", "anemia", "arthritis", "kidney disease", "migraine", "tuberculosis", "malaria", "dengue", "thyroid disorders", "obesity", "osteoporosis"],
    "topics": ["symptoms", "causes", "risk factors", "warning signs", "complications", "prevention", "diagnosis", "management", "treatment options", "recovery", "when to seek medical help", "lifestyle changes", "how it affects the body"],
    "medicines": ["paracetamol", "ibuprofen", "metformin", "insulin", "amlodipine", "losartan", "atorvastatin", "amoxicillin", "inhalers", "antibiotics", "antihistamines", "blood thinners", "oral rehydration solution", "vaccines"],
    "hospital": ["hospital services", "emergency department", "outpatient department", "intensive care unit", "pharmacy", "laboratory tests", "radiology department", "health checkups", "infection control", "patient discharge process", "appointment process", "hospital admission process"],
    "general": ["healthy diet", "balanced nutrition", "hydration", "physical activity", "sleep health", "blood pressure", "blood glucose", "cholesterol", "first aid", "vaccination", "medical terminology", "patient safety"],
}
CHAT = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "how is it going", "thanks", "thank you", "thank you so much", "bye", "goodbye", "see you later", "who are you", "what is your name", "introduce yourself", "what can you do", "how can you help me", "what are your capabilities", "help me get started", "how do I use this chatbot", "explain how this application works", "are you an AI assistant", "are you a healthcare chatbot", "can you help me", "tell me about yourself", "nice to meet you", "I need some help", "can you assist me"
]
INVALID = [
    "write a poem about the ocean", "who won the cricket match", "what is the weather tomorrow", "book a flight to Mumbai", "recommend a gaming laptop", "write Java code for sorting", "solve this algebra equation", "tell me a joke about politics", "what is the latest movie", "give me stock market predictions", "hack an email account", "design a logo for my shop", "what is the capital of France", "explain quantum computing", "create a gaming username", "find celebrity gossip", "plan a holiday in Goa", "translate this paragraph into French", "write a birthday message", "how do I build a car engine"
]

def build_rows():
    rows = set()
    def add(text, label):
        text = " ".join(text.split()).strip(" ?.,")
        if len(text) > 4:
            rows.add((text, label))

    # CHAT: direct conversational/identity/help examples.
    for text in CHAT:
        add(text, "chat")
        for prefix in ["please", "hello assistant", "hey there", "can you tell me", "I wanted to ask"]:
            add(f"{prefix} {text}", "chat")

    # KNOWLEDGE: medical explanations, disease and medicine information,
    # hospital processes and general healthcare education.
    for disease in KNOWLEDGE["diseases"]:
        for topic in KNOWLEDGE["topics"]:
            for template in [
                f"what are the {topic} of {disease}",
                f"explain {topic} related to {disease}",
                f"tell me about {disease} and its {topic}",
                f"I want to understand {topic} for {disease}",
            ]:
                add(template, "knowledge")
    for medicine in KNOWLEDGE["medicines"]:
        for template in [
            f"what is {medicine} used for", f"explain {medicine}",
            f"what should I know about {medicine}",
            f"what are the common side effects of {medicine}",
            f"is {medicine} an antibiotic or another type of medicine",
            f"what precautions are important with {medicine}",
        ]:
            add(template, "knowledge")
    for topic in KNOWLEDGE["hospital"] + KNOWLEDGE["general"]:
        for template in [f"explain {topic}", f"what is {topic}", f"tell me about {topic}", f"how does {topic} work", f"why is {topic} important in healthcare"]:
            add(template, "knowledge")

    # ANALYTICS: questions explicitly asking for values, counts, breakdowns,
    # comparisons, rankings, or records from structured hospital datasets.
    for entity in ANALYTICS["entities"]:
        for condition in ANALYTICS["conditions"]:
            for verb in ["count", "show", "find", "list"]:
                add(f"{verb} the {entity} {condition}", "analytics")
                add(f"how many {entity} are {condition}", "analytics")
        for metric in ANALYTICS["metrics"]:
            add(f"what is the {metric} number of {entity}", "analytics")
            add(f"calculate the {metric} for {entity}", "analytics")
            add(f"give me the {metric} of {entity}", "analytics")
        for timeframe in ANALYTICS["timeframes"]:
            add(f"show {entity} {timeframe}", "analytics")
            add(f"give me the number of {entity} {timeframe}", "analytics")
    for group in ANALYTICS["groups"]:
        for entity in ANALYTICS["entities"]:
            add(f"break down {entity} by {group}", "analytics")
            add(f"show the distribution of {entity} by {group}", "analytics")
            add(f"compare {entity} across {group}", "analytics")
    for condition in ANALYTICS["conditions"]:
        for entity in ["patients", "visits", "prescriptions"]:
            add(f"how many {entity} {condition}", "analytics")
            add(f"show records for {entity} {condition}", "analytics")

    for text in INVALID:
        add(text, "invalid")
        add("please " + text, "invalid")

    # Shuffle deterministically. Avoid a fixed ordering bias during training.
    result = list(rows)
    RNG.shuffle(result)
    return result

if __name__ == "__main__":
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)
    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"Created {len(rows):,} unique examples: {OUT}")
    for label, count in sorted(counts.items()):
        print(f"  {label:10s}: {count:,}")
    print("NOTE: Template-generated data is a starter augmentation set, not a substitute for real labelled queries.")
