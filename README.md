# StudyBot

AI study assistant — upload PDF, ask questions, get taught using RAG

## Tech stack

- FastAPI
- FAISS
- sentence-transformers
- Groq LLaMA 3
- React + Vite + Tailwind

## Setup instructions

### Backend
- `pip install -r requirements.txt`
- add `.env` with `GROQ_API_KEY`
- `uvicorn main:app --reload`

### Frontend
- `npm install`
- `npm run dev`

## How it works

PDF → extract → chunk → embed → FAISS → query → Groq → answer
