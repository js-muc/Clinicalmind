# classifier.py
def classify_query(question):
    q = question.lower()
        # 🆔 exact patient lookup
    if "hsp" in q or "patient id" in q:
        return "lookup"

    # 🔥 reasoning / risk
    if any(w in q for w in [
        "risk",
        "danger",
        "complication",
        "severe",
        "critical",
        "why"
    ]):
        return "reasoning"

    # 🔢 numeric / threshold
    if any(w in q for w in [
        "above",
        "below",
        "greater",
        "less",
        ">",
        "<"
    ]):
        return "numeric"

    # 📊 aggregation / listing
    if any(w in q for w in [
        "how many",
        "count",
        "most common",
        "top",
        "all",
        "list",
        "show",
        "which patients",
        "patients",
        "patients with",
        "patients diagnosed",
        "diagnosed with",
        "who has",
        "who had"
    ]):
       return "aggregation"
        

    # 🩺 symptom search
    return "symptom"


def detect_intent(question):
    q = question.lower()

    if "patient" in q or "diagnosis" in q:
        return "medical"

    elif "policy" in q or "rule" in q:
        return "policy"

    elif "contract" in q or "clause" in q:
        return "legal"

    return "general"


def is_aggregation_query(question):
    q = question.lower()

    keywords = [
        "how many",
        "count",
        "number",
        "list",
        "all",
        "find all",
        "who",
        "which",
        "show"
    ]

    return any(k in q for k in keywords)


def is_frequency_query(question):
    q = question.lower()

    return any(word in q for word in [
        "most common",
        "most frequent",
        "top",
        "highest"
    ])


def is_reasoning_query(question):
    q = question.lower()

    keywords = [
        "check",
        "verify",
        "match",
        "consistent",
        "correct",
        "validate"
    ]

    numeric_keywords = [
        "above",
        "below",
        "greater",
        "less",
        ">",
        "<"
    ]

    return any(
        k in q
        for k in keywords + numeric_keywords
    )
