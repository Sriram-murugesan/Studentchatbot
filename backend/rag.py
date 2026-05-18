import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory session store
sessions = {}  # session_id → {"index": faiss_index, "chunks": [...]}

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, size=500, overlap=50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_bytes: bytes, session_id: str) -> int:
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text)

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]  # 384 for MiniLM

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    sessions[session_id] = {"index": index, "chunks": chunks}
    return len(chunks)

def ask_question(question: str, session_id: str) -> str:
    if session_id not in sessions:
        return "Session not found. Please upload your book again."

    session = sessions[session_id]
    index = session["index"]
    chunks = session["chunks"]

    query_embedding = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=4)

    context = "\n\n".join(chunks[i] for i in indices[0] if i < len(chunks))

    prompt = (
        "You are an expert study assistant and teacher.\n"
        "A student has uploaded their textbook and asked a question.\n\n"
        "Your job is to TEACH, not just answer. Follow this exact structure:\n\n"
        "---\n\n"
        "STEP 1 - DEFINE (2-3 lines max)\n"
        "Give a simple, clear definition of what the student is asking about.\n"
        "You can use your own knowledge for this part.\n\n"
        "STEP 2 - CONNECT TO THE BOOK\n"
        "Find the relevant section in the book context below.\n"
        "Quote or reference what the book says about it.\n\n"
        "STEP 3 - TEACH COMPLETELY\n"
        "Now explain it fully using the book content.\n"
        "Use simple language. If there are steps or processes, explain them in order.\n"
        "Use examples from the book if available.\n\n"
        "---\n\n"
        "RULES:\n"
        "- Never just copy-paste from the book. Always explain in teaching style.\n"
        "- If the concept is not in the book at all, say exactly: "
        "I couldn't find that in your book. "
        "Then give a brief definition from your knowledge only.\n"
        "- Keep the flow natural like a teacher talking to a student.\n"
        "- Use markdown formatting. Use ## for step headers, **bold** for key terms, and short paragraphs.\n\n"
        "---\n\n"
        "FEW-SHOT EXAMPLES:\n\n"
        "Example 1:\n"
        "Question: What is photosynthesis?\n"
        "Book context: Photosynthesis is the process by which plants use sunlight, "
        "water and CO2 to produce glucose and oxygen.\n\n"
        "Answer:\n"
        "Photosynthesis is how living organisms mainly plants convert light energy "
        "into chemical energy they can use as food.\n\n"
        "Your book explains it clearly: plants take in sunlight, water, and carbon "
        "dioxide, and through a series of chemical reactions, produce glucose and "
        "oxygen as a byproduct.\n\n"
        "Think of it like a kitchen. The plant is the chef. Sunlight is the stove, "
        "water and CO2 are the ingredients, and glucose is the meal being cooked. "
        "The oxygen you breathe is just the steam escaping. Your book covers the "
        "inputs and outputs of this process.\n\n"
        "---\n\n"
        "Example 2:\n"
        "Question: What is Newton's second law?\n"
        "Book context: F = ma. The acceleration of an object depends on "
        "the net force acting on it and its mass.\n\n"
        "Answer:\n"
        "Newton's second law describes the relationship between force, mass, and "
        "acceleration. It tells us how much an object will accelerate when a force "
        "is applied to it.\n\n"
        "According to your book, this is expressed as F = ma, meaning force equals "
        "mass times acceleration.\n\n"
        "So if you push a heavy object and a light object with the same force, the "
        "lighter one accelerates faster. Double the force, double the acceleration. "
        "Double the mass, half the acceleration. It is the foundation of how we "
        "calculate motion in physics.\n\n"
        "---\n\n"
        "Now answer this student's question using the same teaching style.\n\n"
        "Think through this before answering:\n"
        "1. What is the student actually asking?\n"
        "2. What relevant information is in the book context?\n"
        "3. How do I explain this like a teacher, not a search engine?\n\n"
        "Book context:\n"
        + context +
        "\n\nStudent's question: " + question +
        "\n\nYour answer:"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content