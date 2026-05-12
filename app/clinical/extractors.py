
# extractors.py

import re


# -------------------------------
# 🔧 NORMALIZE FIELD NAMES
# -------------------------------
def normalize_key(key):

    key = key.lower().strip()

    replacements = {
        "blood pressure": "blood_pressure",
        "heart rate": "heart_rate",
        "patient id": "entity",
        "temperature": "temperature"
    }

    return replacements.get(
        key,
        key.replace(" ", "_")
    )


# -------------------------------
# 🔢 GENERIC NUMERIC EXTRACTION
# -------------------------------
def extract_numeric_fields(text):

    patterns = [
        r"([A-Za-z\s]+):\s*(\d+\.?\d*)",
        r"([A-Za-z\s]+):\s*([\d/]+\s*mmHg)",
        r"([A-Za-z\s]+):\s*(\d+\s*bpm)"
    ]

    results = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        for key, value in matches:

            key = normalize_key(key)

            results.append(
                (key, value.strip())
            )

    return results


# -------------------------------
# 🧠 MAIN STRUCTURED EXTRACTION
# -------------------------------
def extract_fields_from_chunk(chunk):

    fields = {}

    text = chunk

    # -------------------------------
    # 🆔 PATIENT ID
    # -------------------------------
    id_match = re.search(
        r"Patient ID:\s*([A-Z0-9]+)",
        text
    )

    if id_match:
        fields["entity"] = id_match.group(1)

    # -------------------------------
    # 👤 NAME
    # -------------------------------
    name_match = re.search(
        r"Name:\s*([A-Za-z\s]+?)\s*Age",
        text
    )

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
    # 🩺 SYMPTOMS
    # -------------------------------
    symptoms_match = re.search(
        r"Chief Complaint:\s*([^.]+)",
        text
    )

    if symptoms_match:

        symptoms = symptoms_match.group(1).lower()

        symptoms = re.sub(
            r"reported.*",
            "",
            symptoms
        )

        fields["symptoms"] = symptoms.strip()

        print(
            "SYMPTOMS:",
            fields.get("symptoms")
        )

    # -------------------------------
    # 📄 PAGE
    # -------------------------------
    page_match = re.search(
        r"Page\s*(\d+)",
        text
    )

    if page_match:
        fields["page"] = page_match.group(1)

    # -------------------------------
    # 🩺 DIAGNOSIS
    # -------------------------------
    diagnosis_match = re.search(
        r"Diagnosis:\s*(.+?)(?:Treatment Plan:|Medication:|Advice:|$)",
        text,
        re.IGNORECASE
    )

    if diagnosis_match:
        fields["diagnosis"] = diagnosis_match.group(1).strip()

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
    # ACTION
    # -------------------------------
    if "Medication:" in text:

        part = text.split("Medication:")[1]

        stop_words = [
            "Advice:",
            "Follow-up:",
            "Diagnosis:"
        ]

        for stop in stop_words:

            if stop in part:
                part = part.split(stop)[0]

        value = part.strip()

        value = re.sub(
            r"[-\s]+$",
            "",
            value
        )

        fields["action"] = value

    elif "Treatment Plan:" in text:

        part = text.split("Treatment Plan:")[1]

        stop_words = [
            "Medication:",
            "Advice:",
            "Follow-up:"
        ]

        for stop in stop_words:

            if stop in part:
                part = part.split(stop)[0]

        fields["action"] = part.strip()

    # -------------------------------
    # 🔢 GENERIC NUMERIC FIELDS
    # -------------------------------
    numeric_fields = extract_numeric_fields(text)

    for key, value in numeric_fields:
        fields[key] = value

    return fields

