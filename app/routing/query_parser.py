import re


def extract_target_phrase(question):
    q = question.lower()

    # remove noise
    q = re.sub(r"[^\w\s]", "", q)

    stop_words = {
        "what", "which", "who", "where", "when", "why", "how",
        "many", "count", "number", "total", "all",
        "list", "find", "show", "give",
        "are", "is", "was", "were", "have", "has", "had",
        "the", "a", "an", "and", "of", "to", "in",
        "with", "for", "on", "at",
        "patients", "patient", "names", "name"
    }

    words = q.split()

    ignore_words = {
        "name", "names",
        "list", "show", "give", "find",
        "who", "which",
        "patient", "patients"
    }

    ignore_words.update({
        "presented",
        "diagnosis",
        "have",
        "has",
        "had"

    })

    important = [
        w for w in words
        if w not in stop_words
        and w not in ignore_words
        and len(w) > 3
        and not w.isnumeric()
    ]

    mapped = []

    for w in important:

        if w.startswith("diagnos") or w.startswith("dragon"):
            mapped.append("diagnosis")

        else:
            mapped.append(w)

    return list(set(mapped))


def extract_identifier(question):

    patterns = [
        r"\b[A-Z]{2,10}\d{2,}\b",
        r"\b[A-Z]+[-_]\d{2,}\b",
        r"\b[A-Z]{2,10}[-_]\d{2,}\b",
    ]

    q = question.upper()

    for p in patterns:

        match = re.search(p, q)

        if match:
            return match.group()

    return None