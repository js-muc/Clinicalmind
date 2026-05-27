# symptom_engine.py

import re
import numpy as np

from clinical.normalization import (
    normalize_medical_terms
)

SYMPTOM_WEIGHTS = {
    "difficulty breathing": 3,
    "dyspnea": 3,
    "chest pain": 2,
    "angina": 2,
    "cough": 1,
    "fever": 1,
}

def extract_symptoms_from_question(question):

    q = question.lower()

    q = normalize_medical_terms(q)

    only_flag = "only" in q

    q = re.sub(
        r"(which|patients|had|with|only|what|their|diagnosis|was|were)",
        "",
        q
    )

    parts = re.split(r"\band\b|,", q)

    symptoms = []

    for p in parts:

        p = p.strip()

        p = re.sub(
            r"[^\w\s]",
            "",
            p
        )

        if len(p) < 3:
            continue

        symptoms.append(p)

    return symptoms, only_flag

def semantic_match_score(
    query_words,
    text,
    model
):

    if not text.strip():
        return 0

    text_vec = model.encode([text])[0]

    scores = []

    for word in query_words:

        word_vec = model.encode([word])[0]

        sim = np.dot(
            word_vec,
            text_vec
        ) / (

            np.linalg.norm(word_vec)
            *
            np.linalg.norm(text_vec)

        )

        scores.append(sim)

    return np.mean(scores)

def strict_symptom_match(
    query_symptoms,
    record_parts,
    model
):

    matches = []

    record_parts = [
        p.strip()
        for p in record_parts
        if len(p.strip()) > 2
    ]

    symptom_vecs = {
        s: model.encode([s])[0]
        for s in query_symptoms
    }

    for symptom in query_symptoms:

        best_score = 0

        for part in record_parts:

            symptom_vec = symptom_vecs[symptom]

            part_vec = model.encode(
                [part]
            )[0]

            similarity = np.dot(
                symptom_vec,
                part_vec

            ) / (

                np.linalg.norm(symptom_vec)
                *
                np.linalg.norm(part_vec)

            )

            if symptom not in part and similarity < 0.7:
                continue

            best_score = max(
                best_score,
                similarity
            )

        matches.append(
            best_score > 0.65
        )

    return any(matches), matches