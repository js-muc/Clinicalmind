# loader.py
import re
import os

from pypdf import PdfReader

from clinical.extractors import (
    extract_fields_from_chunk
)

# -------------------------------
# 🧠 GLOBAL DOCUMENT STORE
# -------------------------------
DOCUMENT_STORE = []


# -------------------------------
# 🔧 PROCESS PDF
# -------------------------------
def process_pdf(file, model):

    reader = PdfReader(file.name)

    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text()

    clean_text = re.sub(
        r"data:text/html.*",
        "",
        full_text
    )

    clean_text = re.sub(
        r"<.*?>",
        "",
        clean_text
    )

    clean_text = re.sub(
        r"\s+",
        " ",
        clean_text
    ).strip()

    chunks = []
    metadata = []

    records = re.split(
        r'(?=Page \d+)',
        clean_text
    )

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
def load_document(file, model):

    for doc in DOCUMENT_STORE:

        if doc["name"] == file.name:
            return

    chunks, embeddings, metadata = process_pdf(
        file,
        model
    )

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

# -------------------------------
# 🔧 GET ALL RECORDS
# -------------------------------
def get_all_records():

    all_records = []

    for doc in DOCUMENT_STORE:
        all_records.extend(doc["records"])

    return all_records
