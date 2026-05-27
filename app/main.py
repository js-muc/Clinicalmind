import gradio as gr
import re
import numpy as np
from pypdf import PdfReader
from handlers.lookup_handler import handle_lookup

from clinical.extractors import (
    normalize_key,
    extract_numeric_fields,
    extract_fields_from_chunk
)


from clinical.reasoning_engine import (
    generic_consistency_check,
    group_fields_semantically,
    extract_target_field
)

from routing.classifier import (
    classify_query,
    detect_intent,
    is_aggregation_query,
    is_frequency_query,
    is_reasoning_query
)

from routing.query_parser import (
    extract_target_phrase,
    extract_identifier
)

from clinical.normalization import (
    normalize_medical_terms
)

from clinical.symptom_engine import (

    extract_symptoms_from_question,

    semantic_match_score,

    strict_symptom_match

)
from clinical.numeric_engine import (
    handle_numeric_reasoning
)

from clinical.aggregation_engine import (
    handle_aggregation
)

from core.config import (

    client,

    model

)

from handlers.reasoning_handler import (

    handle_reasoning

)

from clinical.record_lookup import find_record_by_identifier

from rag.loader import (
    load_document,
    get_all_records,
    DOCUMENT_STORE
)



# -------------------------------
# 🧠 CHAT MEMORY
# -------------------------------
CHAT_HISTORY = []


def build_prompt(context, question):
    intent = detect_intent(question)

    if intent == "medical":
        structure = """
For EACH patient separately, extract:
- Patient Name
- Patient ID
- Page
- Diagnosis
"""

    elif intent == "policy":
        structure = """
For EACH rule or section, extract:
- Section name
- Rule description
- Requirement
"""

    elif intent == "legal":
        structure = """
For EACH clause, extract:
- Clause name
- Summary
- Conditions
"""

    else:
        structure = """
Extract all relevant facts clearly in bullet points.
"""

    return f"""
You are a strict extraction system.

RULES:
1. Use ONLY the provided context
2. DO NOT summarize across items
3. DO NOT combine different entities
4. Extract information EXACTLY as written
5. If missing → say "Not found in document"

{structure}

Context:
{context}

Question:
{question}
"""

        
# -------------------------------
# 🔧 ANSWER FUNCTION
# -------------------------------
def answer_question(file, question):
    if file is None:
        return "Please upload a PDF first."

    load_document(file, model)

    # 🧠 DEBUG QUERY TYPE
    query_type = classify_query(question)
    # =================================
    # EXACT PATIENT LOOKUP
    # =================================
    if query_type == "lookup":
            return handle_lookup(question)

    print("QUESTION:", question)
    print("ROUTING TO:", query_type)
    print("ROUTING TO:", query_type)

    print("\n=== QUERY TYPE ===")
    print(query_type)

    if query_type == "reasoning":

        return handle_reasoning(

            question,

            model

        )

    # -------------------------------
    # 🎯 EXACT IDENTIFIER MATCH (GENERIC)
    # -------------------------------
    identifier = extract_identifier(question)

    if identifier:
        all_records = get_all_records()

        matched_chunks = [
            r["chunk"]
            for r in all_records
            if identifier in r["chunk"]
        ]

        if matched_chunks:
            context = "\n".join(matched_chunks[:3])

            prompt = build_prompt(context, question)

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a document assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        return f"Identifier {identifier} not found in document."

    if query_type == "aggregation":

        return handle_aggregation(
            question
    )
    
    is_comparison = any(word in question.lower() for word in ["more", "compare", "difference", "strict"])

    expanded_question = question
    if is_comparison:
        expanded_question += " policy rules requirement"

    question_embedding = model.encode([expanded_question])

    # -------------------------------
    # SECTION MATCHING
    # -------------------------------
    section_keywords = {
        "IT Security": ["security", "password", "authentication", "device"],
        "Data Privacy": ["data", "privacy"],
        "Remote Work": ["remote"],
        "Finance": ["finance", "expenditure"],
        "Conduct": ["conduct", "behavior"]
    }

    selected_section = None
    for section, keywords in section_keywords.items():
        if any(word in question.lower() for word in keywords):
            selected_section = section
            break

    # -------------------------------
    # MERGE DOCUMENTS
    # -------------------------------
    all_chunks, all_embeddings, all_metadata, all_sources = [], [], [], []

    for doc in DOCUMENT_STORE:
        all_chunks.extend(doc["chunks"])
        all_embeddings.extend(doc["embeddings"])
        all_metadata.extend(doc["metadata"])
        all_sources.extend(doc["sources"])

    all_embeddings = np.array(all_embeddings)

    # -------------------------------
    # FILTER
    # -------------------------------
    filtered_chunks, filtered_embeddings, filtered_sources, filtered_metadata = [], [], [], []

    for chunk, emb, meta, source in zip(all_chunks, all_embeddings, all_metadata, all_sources):
        if is_comparison:
            filtered_chunks.append(chunk)
            filtered_embeddings.append(emb)
            filtered_sources.append(source)
            filtered_metadata.append(meta)
        else:
            if selected_section is None or selected_section.lower() in meta.lower():
                filtered_chunks.append(chunk)
                filtered_embeddings.append(emb)
                filtered_sources.append(source)
                filtered_metadata.append(meta)

    filtered_embeddings = np.array(filtered_embeddings)

    # -------------------------------
    # SIMILARITY
    # -------------------------------
    chunk_norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
    question_norm = np.linalg.norm(question_embedding, axis=1, keepdims=True)

    normalized_chunks = filtered_embeddings / chunk_norms
    normalized_question = question_embedding / question_norm

    similarity_scores = np.dot(normalized_chunks, normalized_question.T).flatten()

    k = 5

    # dynamic threshold
    threshold = 0.1 if is_comparison else 0.2

    top_indices = similarity_scores.argsort()[::-1]

    top_chunks = []
    top_sources = []
    selected = set()

    # -------------------------------
    # THRESHOLD + FALLBACK RETRIEVAL
    # -------------------------------
    for idx in top_indices:
        score = similarity_scores[idx]
        chunk = filtered_chunks[idx]

        if score >= threshold:
            if chunk not in selected:
                top_chunks.append(chunk)
                top_sources.append(filtered_sources[idx])
                selected.add(chunk)

        if len(top_chunks) == k:
            break

    # fallback (prevents incompleteness)
    if len(top_chunks) < 2:
        top_chunks = []
        top_sources = []
        selected = set()

        for idx in top_indices:
            chunk = filtered_chunks[idx]

            if chunk not in selected:
                top_chunks.append(chunk)
                top_sources.append(filtered_sources[idx])
                selected.add(chunk)

            if len(top_chunks) == k:
                break

    # -------------------------------
    # COMPARISON COVERAGE FIX
    # -------------------------------
    #if is_comparison:
        #section_map = {}

        #for idx in top_indices:
            #meta = filtered_metadata[idx]
            #if meta not in section_map:
                #section_map[meta] = []
            #section_map[meta].append(idx)

        #for section, indices in section_map.items():
            #for i in indices[:2]:
                #chunk = filtered_chunks[i]
                #if chunk not in selected:
                    #top_chunks.append(chunk)
                    #top_sources.append(filtered_sources[i])
                    #selected.add(chunk)

    #if not top_chunks:
        #return "No relevant information found in document."

    #context = "\n".join(top_chunks)

    # -------------------------------
    # GENERATION
    # -------------------------------
    #prompt = build_prompt(context, question)

    #CHAT_HISTORY.append({"role": "user", "content": question})

    #response = client.chat.completions.create(
        #model="gpt-4.1-mini",
        #messages=[
            #{"role": "system", "content": "You are a document assistant."},
            #*CHAT_HISTORY,
            #{"role": "user", "content": prompt}
        #]
    #)

    #answer = response.choices[0].message.content

    #CHAT_HISTORY.append({"role": "assistant", "content": answer})
    #CHAT_HISTORY[:] = CHAT_HISTORY[-6:]

    #unique_sources = list(set(top_sources))

    #return answer + "\n\n📚 Sources:\n" + "\n".join(unique_sources)


# -------------------------------
# UI
# -------------------------------
app = gr.Interface(
    fn=answer_question,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs=gr.Textbox(
        lines=25,
        max_lines=40,
        label="ClinicalMind Output"
    ),
    title="📄 Ask Your PDF ClinicalMind",
    description="Upload a document and ask questions"
)

if __name__ == "__main__":
    app.launch()