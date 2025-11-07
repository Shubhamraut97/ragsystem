import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Test chat
model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content("Say 'Gemini API is working!'")
print(response.text)

# Test embeddings
# embedding_model = genai.GenerativeModel("models/embedding-001")
# result = genai.embed_content(
#     model="models/embedding-001", content="This is a test sentence"
# )
result = genai.embed_content(
    model="gemini-embedding-001",
    content="This is a test sentence",  
)
print(f"Embedding dimension: {len(result['embedding'])}")
