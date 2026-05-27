# config.py

import os

from dotenv import load_dotenv

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer
)

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

client = None

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    print(
        "✅ OpenAI client initialized"
    )

else:

    print(
        "⚠️ No OpenAI API key found"
    )


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)