# lookup_handler.py

from routing.query_parser import extract_identifier

from clinical.record_lookup import find_record_by_identifier

from rag.loader import get_all_records


# -------------------------------
# 🔎 HANDLE PATIENT LOOKUP
# -------------------------------
def handle_lookup(question):

    all_records = get_all_records()

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