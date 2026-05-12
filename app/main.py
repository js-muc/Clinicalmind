import gradio as gr
import re
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

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

from clinical.record_lookup import find_record_by_identifier

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized")

else:
    print("⚠️ No OpenAI API key found — running local mode")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 🧠 MULTI-DOCUMENT STORE
# -------------------------------
DOCUMENT_STORE = []

# -------------------------------
# 🧠 CHAT MEMORY
# -------------------------------
CHAT_HISTORY = []


# -------------------------------
# 🔧 PROCESS PDF
# -------------------------------
def process_pdf(file):
    reader = PdfReader(file.name)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    clean_text = re.sub(r"data:text/html.*", "", full_text)
    clean_text = re.sub(r"<.*?>", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    chunks = []
    metadata = []

    records = re.split(r'(?=Page \d+)', clean_text)

    for r in records:
        r = r.strip()
        if not r:
            continue

        chunk = "Page " + r
        chunks.append(chunk)

        metadata.append("record")

    embeddings = model.encode(chunks)

    return chunks, embeddings, metadata


# -------------------------------
# 🔧 LOAD DOCUMENT
# -------------------------------
def load_document(file):
    for doc in DOCUMENT_STORE:
        if doc["name"] == file.name:
            return

    chunks, embeddings, metadata = process_pdf(file)

    records = []

    for i, chunk in enumerate(chunks):

        fields = extract_fields_from_chunk(chunk)

        records.append({
            "chunk": chunk,
            "embedding": embeddings[i],
            "metadata": metadata[i],
            "source": os.path.basename(file.name),
            "page": fields.get("page", "Unknown"),
            "fields": fields
        })

    DOCUMENT_STORE.append({
        "name": file.name,
        "records": records
    })

def extract_fields_from_chunk(chunk):
    fields = {}

    text = chunk

   
    # -------------------------------
    # 🆔 ENTITY
    # -------------------------------
    id_match = re.search(r"Patient ID:\s*([A-Z0-9]+)", text)

    if id_match:
        fields["entity"] = id_match.group(1)

    # -------------------------------
    # 👤 NAME (independent)
    # -------------------------------
    name_match = re.search(r"Name:\s*([A-Za-z\s]+?)\s*Age", text)

    if name_match:
        fields["name"] = name_match.group(1).strip()
    

        # -------------------------------
    # 🎂 AGE
    # -------------------------------
    age_match = re.search(
        r"Age:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if age_match:
        fields["age"] = age_match.group(1)


        # -------------------------------
    # 🚻 GENDER
    # -------------------------------
    gender_match = re.search(
        r"Gender:\s*(Male|Female)",
        text,
        re.IGNORECASE
    )

    if gender_match:
        fields["gender"] = gender_match.group(1)    
    # -------------------------------
    # 🩺 SYMPTOMS (Chief Complaint)  
    # -------------------------------
    symptoms_match = re.search(
        r"Chief Complaint:\s*([^.]+)",
        text
    )

    if symptoms_match:
        symptoms = symptoms_match.group(1).lower()

        # remove noise phrases
        symptoms = re.sub(r"reported.*", "", symptoms)

        fields["symptoms"] = symptoms.strip()  

        print("SYMPTOMS:", fields.get("symptoms"))  

    # -------------------------------
    # 📄 PAGE (independent)
    # -------------------------------
    page_match = re.search(r"Page\s*(\d+)", text)

    if page_match:
        fields["page"] = page_match.group(1)
    diagnosis_match = re.search(
        r"Diagnosis:\s*([A-Za-z\s]+)",
        text,
        re.IGNORECASE
    )

    diagnosis_match = re.search(
        r"Diagnosis:\s*(.+?)(?:Treatment Plan:|Medication:|Advice:|$)",
        text,
        re.IGNORECASE
    )

    if diagnosis_match:
        fields["diagnosis"] = diagnosis_match.group(1).strip()

        # -------------------------------
    # 💊 MEDICATION
    # -------------------------------
    medication_match = re.search(
        r"Medication:\s*(.+?)(?:Advice:|Follow-up:|$)",
        text,
        re.IGNORECASE
    )

    if medication_match:
        fields["medication"] = medication_match.group(1).strip()   
    # -------------------------------
    # ACTION (treatment / medication)
    # -------------------------------
    if "Medication:" in text:
        part = text.split("Medication:")[1]

        stop_words = ["Advice:", "Follow-up:", "Diagnosis:"]
        for stop in stop_words:
            if stop in part:
                part = part.split(stop)[0]

        value = part.strip()

        # remove trailing junk like "-"
        value = re.sub(r"[-\s]+$", "", value)

        fields["action"] = value

    elif "Treatment Plan:" in text:
        part = text.split("Treatment Plan:")[1]

        stop_words = ["Medication:", "Advice:", "Follow-up:"]
        for stop in stop_words:
            if stop in part:
                part = part.split(stop)[0]

        fields["action"] = part.strip()

        # -------------------------------
        # 🔢 ADD GENERIC NUMERIC FIELDS
        # -------------------------------
    numeric_fields = extract_numeric_fields(text)
        # -------------------------------
    # 🩸 BLOOD PRESSURE
    # -------------------------------
    bp_match = re.search(
        r"Blood Pressure:\s*([\d/]+\s*mmHg)",
        text,
        re.IGNORECASE
    )

    if bp_match:
        fields["blood_pressure"] = bp_match.group(1).strip()

    
        # -------------------------------
    # ❤️ HEART RATE
    # -------------------------------
    hr_match = re.search(
        r"Heart Rate:\s*(\d+\s*bpm)",
        text,
        re.IGNORECASE
    )

    if hr_match:
        fields["heart_rate"] = hr_match.group(1).strip()

    for key, value in numeric_fields:
        fields[key] = value

    return fields

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

def semantic_match_score(query_words, text):
    if not text.strip():
        return 0

    text_vec = model.encode([text])[0]

    scores = []

    for word in query_words:
        word_vec = model.encode([word])[0]

        sim = np.dot(word_vec, text_vec) / (
            np.linalg.norm(word_vec) * np.linalg.norm(text_vec)
        )

        scores.append(sim)

    return np.mean(scores)


def normalize_medical_terms(text):
    text = text.lower()

    mapping = {
        "shortness of breath": "dyspnea",
        "chest pain": "angina",
        "fever": "pyrexia",
        "high fever": "pyrexia",
        "burning urination": "dysuria"
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    return text
# 🧠 CLINICAL WEIGHTS (ADD HERE)
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

    # detect "only"
    only_flag = "only" in q

    # remove noise
    q = re.sub(r"(which|patients|had|with|only|what|their|diagnosis|was|were)", "", q)

    parts = re.split(r"\band\b|,", q)

    symptoms = []

    for p in parts:
        p = p.strip()
        p = re.sub(r"[^\w\s]", "", p)

        if len(p) < 3:
            continue

        symptoms.append(p)

    return symptoms, only_flag
def strict_symptom_match(query_symptoms, record_parts):
    matches = []

    record_parts = [p.strip() for p in record_parts if len(p.strip()) > 2]
    symptom_vecs = {s: model.encode([s])[0] for s in query_symptoms}
    
    for symptom in query_symptoms:
        best_score = 0

        for part in record_parts:
            symptom_vec = symptom_vecs[symptom]

            # ✅ ADD THIS LINE (CRITICAL FIX)
            part_vec = model.encode([part])[0]

            similarity = np.dot(symptom_vec, part_vec) / (
               np.linalg.norm(symptom_vec) * np.linalg.norm(part_vec)
            )
            # 🚫 HARD CLINICAL FILTER
            if symptom not in part and similarity < 0.7:
                continue

            best_score = max(best_score, similarity)

        if best_score > 0.65:   # slightly higher threshold
            matches.append(True)
        else:
            matches.append(False)

    total_score = 0
    max_possible = 0

    for i, symptom in enumerate(query_symptoms):

        weight = SYMPTOM_WEIGHTS.get(symptom, 1)
        max_possible += weight

        if matches[i]:
            total_score += weight

    confidence = total_score / max_possible if max_possible > 0 else 0

    # 🚨 PENALIZE EXTRA SYMPTOMS (NEW LOGIC)
    extra_count = 0

    for part in record_parts:
        match = False
        for qs in query_symptoms:
            if qs in part or part in qs:
               match = True
               break

        if not match:
            extra_count += 1

    # apply penalty
    if extra_count > 0:
        penalty = min(0.1 * extra_count, 0.3)
        confidence = confidence - penalty
        confidence = max(confidence, 0)

    # ✅ ALWAYS RUN THIS
    match_count = sum(matches)

    if match_count >= 1 and confidence >= 0.4:
        return True, matches, confidence

    return False, matches, confidence  
        
# -------------------------------
# 🔧 ANSWER FUNCTION
# -------------------------------
def answer_question(file, question):
    if file is None:
        return "Please upload a PDF first."

    load_document(file)

    # 🧠 DEBUG QUERY TYPE
    query_type = classify_query(question)
        # =================================
        # EXACT PATIENT LOOKUP
        # =================================

    if query_type == "lookup":
        all_records = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])    

        identifier = extract_identifier(question)

        if not identifier:
            return "No patient identifier found."

        matched_record = find_record_by_identifier(
            all_records,
            identifier
        )

        if not matched_record:
            return "Patient record not found."
        fields = matched_record["fields"]

        result = f"""
Patient Name: {fields.get('name', 'Unknown')}
Patient ID: {fields.get('entity', 'N/A')}
Age: {fields.get('age', 'N/A')}
Gender: {fields.get('gender', 'N/A')}

Diagnosis:
{fields.get('diagnosis', 'N/A')}

Symptoms:
{fields.get('symptoms', 'N/A')}

Temperature:
{fields.get('temperature', 'N/A')}

Blood Pressure:
{fields.get('blood_pressure', 'N/A')}

Heart Rate:
{fields.get('heart_rate', 'N/A')}

Medication:
{fields.get('medication', 'N/A')}

Page:
{matched_record.get('page', 'N/A')}

Source Document:
{matched_record.get('source', 'Unknown')}
"""

        return result
    print("QUESTION:", question)
    print("ROUTING TO:", query_type)
    print("ROUTING TO:", query_type)

    print("\n=== QUERY TYPE ===")
    print(query_type)

    # -------------------------------
    # 🧠 REASONING MODE (GENERIC)
    # -------------------------------
    if query_type == "reasoning":

        all_records = []
        all_chunks = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])

        for record in all_records:
            all_chunks.append(record["chunk"])

        results = []

        question_embedding = model.encode([question])

        chunk_embeddings = []

        for record in all_records:
            chunk_embeddings.append(record["embedding"])

        chunk_embeddings = np.array(chunk_embeddings)

        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        question_norm = np.linalg.norm(question_embedding, axis=1, keepdims=True)

        normalized_chunks = chunk_embeddings / chunk_norms
        normalized_question = question_embedding / question_norm

        scores = np.dot(normalized_chunks, normalized_question.T).flatten()

        top_indices = [i for i in scores.argsort()[::-1] if scores[i] > 0.2][:10]

        relevant_chunks = [all_records[i]["chunk"] for i in top_indices]

        for chunk in relevant_chunks[:3]:  # limit for safety
            print("\n--- CHUNK DEBUG ---\n", chunk[:200])
            fields = extract_fields_from_chunk(chunk)

            if not fields:
                continue

            validation = generic_consistency_check(fields, question, model)

            # ❗ SKIP NON-MATCHING RECORDS
            if not validation:
                continue

            groups = group_fields_semantically(fields)

            results.append(f"""
            Entity:
            {groups["entity"]}

            Attribute:
            {groups["attribute"]}

            Action:
            {groups["action"]}

            Reasoning:
            {chr(10).join(validation)}
            """)

        return "\n\n".join(results)

    # -------------------------------
    # 🔢 NUMERIC REASONING ENGINE
    # -------------------------------
    if query_type == "numeric":

        all_records = []
        all_chunks = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])

        for record in all_records:
            all_chunks.append(record["chunk"])

        results = []

        # -------------------------------
        # EXTRACT THRESHOLD
        # -------------------------------
        threshold_match = re.search(r"(\d+\.?\d*)", question)

        if not threshold_match:
           return "No numeric threshold found."

        threshold = float(threshold_match.group(1))

        # -------------------------------
        # DETECT TARGET FIELD
        # -------------------------------
        target_field = extract_target_field(question)

        print("\nTARGET FIELD:", target_field)
        print("THRESHOLD:", threshold)

        # -------------------------------
        # SCAN ALL RECORDS
        # -------------------------------
        for record in all_records:

            chunk = record["chunk"]
            source = record["source"]
            page = record["page"]

            fields = extract_fields_from_chunk(chunk)

            if not fields:
                continue

            print("\n--- NUMERIC RECORD ---")
            print(fields)

            matched_reasoning = generic_consistency_check(
                fields,
                question,
                model
            )
            if not matched_reasoning:
                continue
            

            # extract numeric value safely
            value = fields.get(target_field)

            result = f"""
    Patient Name: {fields.get('name', 'Unknown')}
    Source File: {source}
    Page: {page}
    Patient ID: {fields.get('entity', 'N/A')}
    Page: {fields.get('page', 'N/A')}
    Diagnosis: {fields.get('diagnosis', 'N/A')}
    {target_field.title()}: {value}
    Reasoning:
    {chr(10).join(matched_reasoning)}
    """

            results.append(result)

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
        if not results:
            return "No matching numeric records found."

        return "\n\n".join(results)    

    # -------------------------------
    # 🎯 EXACT IDENTIFIER MATCH (GENERIC)
    # -------------------------------
    identifier = extract_identifier(question)

    if identifier:
        all_records = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])

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

    # -------------------------------
    # 🚀 AGGREGATION HANDLING (NEW)
    # -------------------------------
    # -------------------------------
    # 📊 FREQUENCY ANALYSIS (NEW)
    # -------------------------------
    if query_type == "aggregation" and "most common" in question.lower():

        all_records = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])

        diagnosis_counts = {}

        for record in all_records:

            chunk = record["chunk"]
            source = record["source"]
            page = record["page"]
            fields = extract_fields_from_chunk(chunk)

            diagnosis = fields.get("diagnosis")
            if not diagnosis:
                continue

            diagnosis = diagnosis.lower().strip()

            # normalize
            diagnosis = diagnosis.replace("treatment plan", "").strip()

            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1

        if not diagnosis_counts:
            return "No diagnoses found."

        # find most common
        most_common = max(diagnosis_counts, key=diagnosis_counts.get)
        count = diagnosis_counts[most_common]

        return f"Most common disease: {most_common.title()} (Total: {count})"
    if query_type == "aggregation":

        # merge all chunks
        all_records = []

        for doc in DOCUMENT_STORE:
            all_records.extend(doc["records"])
            print("\n=== TOTAL RECORDS ===", len(all_records))

        

        

        matched_chunks = []
        matched_records = []

       

        for record in all_records:

            chunk = record["chunk"]
            source = record["source"]
            page = record["page"]
            fields = extract_fields_from_chunk(chunk)
            if "diagnosis" in fields:
                print("\n--- RECORD FOUND ---")
                print("ENTITY:", fields.get("entity"))
                print("NAME:", fields.get("name"))
                print("DIAGNOSIS:", fields.get("diagnosis"))

            # ALWAYS build both
            combined_fields = " ".join([
                f"{k} {v}".lower() for k, v in fields.items()
            ]) if fields else ""

            combined_full = chunk.lower()

            # 🔥 NEW: normalize raw text too
            combined_full = combined_full.replace("performed diagnosis", "diagnosis")
            # check if ALL important words exist
            # ✅ STRICT MATCH: ONLY CHECK ACTUAL FIELD VALUE
            

            

            # 🔥 combine BOTH structured + raw text
            symptoms = fields.get("symptoms", "").lower()
            diagnosis = fields.get("diagnosis", "").lower()
            diagnosis = normalize_medical_terms(diagnosis)

            search_space = normalize_medical_terms(symptoms + " " + diagnosis)
            
            # -------------------------------
            # 🧠 CLEAN AGGREGATION FILTER
            # -------------------------------

            target_words = extract_target_phrase(question)
 
            print("\nTARGET WORDS:", target_words)
            print("DIAGNOSIS:", diagnosis)

            matched = False

            for word in target_words:

                if word in diagnosis:
                    matched = True
                    break

                symptom_parts = [
                    s.strip()
                    for s in symptoms.split(",")
                ]

                for part in symptom_parts:
                    if word == part:
                        matched = True
                        break

            if matched:
                matched_chunks.append((chunk, 1.0))
                matched_records.append({
                    "fields": fields,
                    "score": 1.0,
                    "source": source,
                    "page": page
                })

                print("\n=== MATCHED RECORD ===")
                print(fields.get("entity"))
                print(fields.get("diagnosis"))
 
        if not matched_chunks:
            return "No matching records found."

        # ✅ STEP 1: collect unique entities safely
        unique_records = {}

        for item in matched_records:

            rec = item["fields"]
            score = item["score"]
            entity = rec.get("entity")
            name = rec.get("name")

            key = entity if entity else name

            if key and key not in unique_records:
                unique_records[key] = {
                    "fields": rec,
                    "score": score,
                    "source": item["source"],
                    "page": item["page"]
                }
        # ✅ DEBUG OUTSIDE LOOP
        print("\n=== UNIQUE RECORD KEYS ===")
        for k in unique_records:
            print("KEY:", k)

    
        # ✅ STEP 2: build clean output
        result = f"Matching records:\n\n"

        for key, item in unique_records.items():

            rec = item["fields"]

            result += (
                f"- Patient Name: {rec.get('name','Unknown')}\n"
                f"  Patient ID: {rec.get('entity','N/A')}\n"
                f"  Page: {item['page']}\n"
                f"  Diagnosis: {rec.get('diagnosis','N/A')}\n"
                f"  Source File: {item['source']}\n"
                f"  Confidence: {round(item['score'], 2)}\n\n"
            )
        # ✅ STEP 3: correct count
        result += f"\nTotal count: {len(unique_records)}"

        return result
    

    

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
    outputs="text",
    title="📄 Ask Your PDF ClinicalMind",
    description="Upload a document and ask questions"
)

if __name__ == "__main__":
    app.launch()