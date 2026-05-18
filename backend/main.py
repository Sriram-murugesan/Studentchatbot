from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ingest_pdf, ask_question
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # Validate file size (50 MB max)
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB")
    
    session_id = str(uuid.uuid4())
    
    try:
        chunk_count = ingest_pdf(pdf_bytes, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    
    return {"session_id": session_id, "chunks": chunk_count}

@app.post("/ask")
async def ask(req: QuestionRequest):
    if not req.session_id or not req.question:
        raise HTTPException(status_code=400, detail="session_id and question required")
    
    try:
        answer = ask_question(req.question, req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")
    
    return {"answer": answer}
@app.get("/")
def home():
    return {"message": "Student Chatbot API Running"}