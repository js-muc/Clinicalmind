# normalization.py

def normalize_medical_terms(text):

    text = text.lower()

    mapping = {
        "shortness of breath": "dyspnea",
        "chest pain": "angina",
        "burning urination": "dysuria"
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    return text