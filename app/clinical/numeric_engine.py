# numeric_engine.py
import re

from rag.loader import get_all_records

from clinical.extractors import (
    extract_fields_from_chunk
)

from clinical.reasoning_engine import (
    generic_consistency_check,
    extract_target_field
)


def handle_numeric_reasoning(
    question,
    model
):

    all_records = get_all_records()

    results = []

    threshold_match = re.search(
        r"(\d+\.?\d*)",
        question
    )

    if not threshold_match:
        return "No numeric threshold found."

    threshold = float(
        threshold_match.group(1)
    )

    target_field = extract_target_field(
        question
    )

    print(
        "\nTARGET FIELD:",
        target_field
    )

    print(
        "THRESHOLD:",
        threshold
    )

    for record in all_records:

        chunk = record["chunk"]

        source = record["source"]

        page = record["page"]

        fields = extract_fields_from_chunk(
            chunk
        )

        if not fields:
            continue

        matched_reasoning = (
            generic_consistency_check(
                fields,
                question,
                model
            )
        )

        if not matched_reasoning:
            continue

        value = fields.get(
            target_field
        )

        result = f"""
Patient Name:
{fields.get('name','Unknown')}

Source File:
{source}

Page:
{page}

Patient ID:
{fields.get('entity','N/A')}

Diagnosis:
{fields.get('diagnosis','N/A')}

{target_field.title()}:
{value}

Reasoning:
{chr(10).join(matched_reasoning)}
"""

        results.append(result)

    if not results:

        return (
            "No matching numeric records found."
        )

    return "\n\n".join(results)
