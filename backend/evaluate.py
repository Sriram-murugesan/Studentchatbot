import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from rag import ingest_pdf, ask_question, sessions, embedder
from test_data import test_cases

load_dotenv()

# ── Step 1 — Upload the PDF ──────────────────────────────
PDF_PATH = "test_book.pdf"

print("Loading PDF...")
with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()

session_id = "eval_session"
chunk_count = ingest_pdf(pdf_bytes, session_id)
print(f"Indexed {chunk_count} chunks\n")

# ── Step 2 — Run each test case through RAG ──────────────
print("Running test cases...\n")

questions = []
answers = []
contexts = []
ground_truths = []

for tc in test_cases:
    question = tc["question"]
    ground_truth = tc["ground_truth"]

    answer = ask_question(question, session_id)

    import numpy as np
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    session = sessions[session_id]
    _, indices = session["index"].search(query_embedding, k=4)
    retrieved_chunks = [
        session["chunks"][i]
        for i in indices[0]
        if i < len(session["chunks"])
    ]

    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved_chunks)
    ground_truths.append(ground_truth)

    print(f"Q: {question}")
    print(f"A: {answer[:150]}...")
    print()

# ── Step 3 — Setup Groq for RAGAS ───────────────────────
print("Setting up Groq for RAGAS...\n")

groq_llm = llm_factory(
    "llama-3.1-8b-instant",
    client=ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
)

ragas_embeddings = RagasHFEmbeddings(model="all-MiniLM-L6-v2")

# ── Step 4 — RAGAS Evaluation ────────────────────────────
print("Running RAGAS evaluation...\n")

metrics = [
    Faithfulness(llm=groq_llm),
    AnswerRelevancy(llm=groq_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=groq_llm),
    ContextRecall(llm=groq_llm),
]

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

result = evaluate(dataset, metrics=metrics)

print("\n=== RAGAS SCORES ===")
print(f"Faithfulness:      {result['faithfulness']:.2f}")
print(f"Answer Relevancy:  {result['answer_relevancy']:.2f}")
print(f"Context Precision: {result['context_precision']:.2f}")
print(f"Context Recall:    {result['context_recall']:.2f}")
print(f"Context Recall:    {result['context_recall']:.2f}")

# ── Step 5 — LLM-as-Judge ────────────────────────────────
print("\n=== LLM-AS-JUDGE SCORES ===")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
total_score = 0

for i, (q, a) in enumerate(zip(questions, answers)):
    judge_prompt = (
        f"You are an expert evaluator. Score this answer from 1 to 5.\n\n"
        f"Question: {q}\n"
        f"Answer: {a}\n\n"
        f"5 = perfectly answers, clear, complete\n"
        f"4 = correct with minor issues\n"
        f"3 = partially answers\n"
        f"2 = mostly irrelevant\n"
        f"1 = wrong or hallucinated\n\n"
        f"Do not favor longer answers.\n\n"
        f"Respond ONLY in this format:\n"
        f"Score: <1-5>\n"
        f"Reason: <one sentence>"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0
    )

    output = response.choices[0].message.content.strip()
    lines = output.split("\n")

    try:
        score = int(lines[0].replace("Score:", "").strip())
        reason = lines[1].replace("Reason:", "").strip()
    except:
        score = 0
        reason = "Could not parse"

    total_score += score
    print(f"Q{i+1}: {q}")
    print(f"     Score: {score}/5 — {reason}\n")

avg_score = total_score / len(questions)
print(f"Average LLM-as-Judge Score: {avg_score:.1f}/5")

# ── Step 6 — Final verdict ───────────────────────────────
print("\n=== FINAL VERDICT ===")
print(f"RAGAS Faithfulness:      {result['faithfulness']:.2f}")
print(f"RAGAS Answer Relevancy:  {result['answer_relevancy']:.2f}")
print(f"RAGAS Context Precision: {result['context_precision']:.2f}")
print(f"RAGAS Context Recall:    {result['context_recall']:.2f}")
print(f"LLM Judge Average:       {avg_score:.1f}/5")

if result['faithfulness'] > 0.7 and avg_score >= 3.5:
    print("\nVerdict: READY TO SHIP ✓")
elif result['faithfulness'] > 0.5:
    print("\nVerdict: NEEDS IMPROVEMENT")
else:
    print("\nVerdict: NOT READY — high hallucination detected")