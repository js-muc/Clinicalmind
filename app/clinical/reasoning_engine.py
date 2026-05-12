# reasoning_engine.py
import re
import numpy as np


def group_fields_semantically(fields):
    groups = {
        "entity": [],
        "attribute": [],
        "action": [],
        "observation": []
    }

    for key, value in fields.items():
        k = key.lower()

        if k == "entity":
            groups["entity"].append((key, value))

        elif any(w in k for w in [
            "diagnosis",
            "condition",
            "status",
            "type"
        ]):
            groups["attribute"].append((key, value))

        elif k == "action":
            groups["action"].append((key, value))

        else:
            groups["observation"].append((key, value))

    return groups


def extract_target_field(question):
    q = question.lower()

    words = q.split()

    ignore = [
        "above",
        "below",
        "greater",
        "less",
        "than",
        "with",
        "find",
        "a",
        "an"
    ]

    target_words = [
        w for w in words
        if w not in ignore
        and not re.match(r"\d+", w)
    ]

    return target_words[-1] if target_words else None


def generic_consistency_check(fields, question, model):

    results = []

    q = question.lower()

    # -------------------------------
    # NUMERIC REASONING
    # -------------------------------
    if any(op in q for op in [
        ">",
        "<",
        "above",
        "below",
        "greater",
        "less"
    ]):

        threshold_match = re.search(r"(\d+\.?\d*)", q)

        if threshold_match:
            threshold = float(threshold_match.group(1))
        else:
            return []

        target_field = extract_target_field(question)

        for key, value in fields.items():

            if target_field and target_field not in key:
                continue

            nums = re.findall(r"\d+\.?\d*", str(value))

            if not nums:
                continue

            val = float(nums[0])

            if ">" in q or "above" in q or "greater" in q:

                if val > threshold:
                    results.append(
                        f"{key}: {val} > {threshold} ✅"
                    )

            elif "<" in q or "below" in q or "less" in q:

                if val < threshold:
                    results.append(
                        f"{key}: {val} < {threshold} ✅"
                    )

        return results

    # -------------------------------
    # SEMANTIC REASONING
    # -------------------------------
    groups = group_fields_semantically(fields)

    attributes = groups["attribute"]
    actions = groups["action"]

    for (k1, v1) in attributes:
        for (k2, v2) in actions:

            vec1 = model.encode([v1])[0]
            vec2 = model.encode([v2])[0]

            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1)
                * np.linalg.norm(vec2)
            )

            if similarity > 0.4:
                results.append(
                    f"{k1} ↔ {k2}: MATCH "
                    f"(score={similarity:.2f})"
                )
            else:
                results.append(
                    f"{k1} ↔ {k2}: MISMATCH "
                    f"(score={similarity:.2f})"
                )

    return results
