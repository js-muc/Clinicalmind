import numpy as np

from rag.loader import (
    get_all_records
)

from clinical.extractors import (
    extract_fields_from_chunk
)

from clinical.reasoning_engine import (

    generic_consistency_check,

    group_fields_semantically

)


def handle_reasoning(

    question,

    model

):

    all_records = get_all_records()

    results = []

    question_embedding = model.encode(

        [question]

    )

    chunk_embeddings = [

        r["embedding"]

        for r in all_records

    ]

    chunk_embeddings = np.array(

        chunk_embeddings

    )

    chunk_norms = np.linalg.norm(

        chunk_embeddings,

        axis=1,

        keepdims=True

    )

    question_norm = np.linalg.norm(

        question_embedding,

        axis=1,

        keepdims=True

    )

    normalized_chunks = (

        chunk_embeddings

        /

        chunk_norms

    )

    normalized_question = (

        question_embedding

        /

        question_norm

    )

    scores = np.dot(

        normalized_chunks,

        normalized_question.T

    ).flatten()

    top_indices = [

        i

        for i in scores.argsort()[::-1]

        if scores[i] > 0.2

    ][:10]

    relevant_chunks = [

        all_records[i]["chunk"]

        for i in top_indices

    ]

    for chunk in relevant_chunks[:3]:

        fields = extract_fields_from_chunk(

            chunk

        )

        if not fields:

            continue

        validation = (

            generic_consistency_check(

                fields,

                question,

                model

            )

        )

        if not validation:

            continue

        groups = (

            group_fields_semantically(

                fields

            )

        )

        results.append(

f"""
Entity:
{groups["entity"]}

Attribute:
{groups["attribute"]}

Action:
{groups["action"]}

Reasoning:
{chr(10).join(validation)}
"""

        )

    if not results:

        return (

            "No reasoning matches found."

        )

    return "\n\n".join(

        results

    )