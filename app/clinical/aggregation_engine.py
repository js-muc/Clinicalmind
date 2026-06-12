# aggregation_engine.py

from rag.loader import (
    get_all_records
)

from routing.query_parser import (
    extract_target_phrase
)

from clinical.extractors import (
    extract_fields_from_chunk
)

from clinical.normalization import (
    normalize_medical_terms
)


def handle_aggregation(
    question
):

    all_records = get_all_records()

    if "most common" in question.lower():

        diagnosis_counts = {}

        for record in all_records:

            fields = extract_fields_from_chunk(
                record["chunk"]
            )

            diagnosis = fields.get(
                "diagnosis"
            )

            if not diagnosis:
                continue

            diagnosis = (
                diagnosis
                .lower()
                .replace(
                    "treatment plan",
                    ""
                )
                .strip()
            )

            diagnosis_counts[
                diagnosis
            ] = (
                diagnosis_counts.get(
                    diagnosis,
                    0
                ) + 1
            )

        if not diagnosis_counts:

            return (
                "No diagnoses found."
            )

        disease = max(
            diagnosis_counts,
            key=diagnosis_counts.get
        )

        total = diagnosis_counts[
            disease
        ]

        return (
            f"Most common disease: "
            f"{disease.title()} "
            f"(Total: {total})"
        )

    matched_records = []

    for record in all_records:

        fields = extract_fields_from_chunk(
            record["chunk"]
        )

        symptoms = normalize_medical_terms(

            fields.get(
                "symptoms",
                ""
            ).lower()

        )

        diagnosis = normalize_medical_terms(

            fields.get(
                "diagnosis",
                ""
            ).lower()

        )
        print("DIAGNOSIS:", diagnosis)

        target_words = (
            extract_target_phrase(
                question
            )
        )

        target_words = [

            normalize_medical_terms(
                word.lower().strip()
            )

            for word in extract_target_phrase(
                question
            )

            if len(word.strip()) > 2
        ]
        print("TARGET WORDS:", target_words)

        symptom_parts = [

            normalize_medical_terms(
                s.strip()
            )

            for s in symptoms.split(",")

        ]

        matched = False

        full_search_space = " ".join(

           symptom_parts +

           [diagnosis]

        ).lower()


        # SINGLE PHRASE QUERY
        if len(target_words) == 1:

            matched = (

                target_words[0]

                in full_search_space

            )

        # MULTI WORD QUERY
        else:

            matched = all(

                word in full_search_space

                for word in target_words

            )
        print(
            "MATCH CHECK:",
            matched,
            diagnosis
        )
        if matched:

            matched_records.append({

                "fields": fields,

                "source":
                record["source"],

                "page":
                record["page"]

            })

    if not matched_records:

        return (
            "No matching records found."
        )

    unique = {}

    for item in matched_records:

        fields = item["fields"]

        entity = (
            fields.get("entity")
            or
            fields.get("name")
        )

        if entity:

            unique[
                entity
            ] = item

    result = (
        "Matching records:\n\n"
    )

    for item in unique.values():

        fields = item["fields"]

        result += (

f"- Patient Name: {fields.get('name')}\n"

f"  Patient ID: {fields.get('entity')}\n"

f"  Page: {item['page']}\n"

f"  Diagnosis: {fields.get('diagnosis')}\n"

f"  Source: {item['source']}\n"

f"  Confidence: 1.0\n\n"

        )

    result += (
        f"\nTotal count: "
        f"{len(unique)}"
    )

    return result                               