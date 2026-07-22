from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List
import re
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","of","in","on",
    "to","for","and","or","but","with","at","by","from","as","that","this",
    "these","those","it","its","what","which","who","whom","when","where",
    "why","how","do","does","did","can","could","will","would","should",
    "shall","may","might","must","not","no","so","than","then","there",
    "their","them","he","she","his","her","you","your","i","we","our","us"
}

FALLBACK_RESPONSE = {
    "answer": "I don't know",
    "citations": [],
    "confidence": 0.0,
    "answerable": False
}

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str = ""
    chunks: List[Chunk] = []

def tokenize(text: str):
    return [w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1]

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=200, content=FALLBACK_RESPONSE)

@app.exception_handler(Exception)
async def general_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content=FALLBACK_RESPONSE)

@app.get("/")
def health():
    return {"status": "ok", "service": "grounded-qa-api"}

@app.post("/grounded-answer")
def grounded_answer(payload: QARequest):
    question = (payload.question or "").strip()
    chunks = payload.chunks or []

    if not question or not chunks:
        return FALLBACK_RESPONSE

    q_keywords = set(tokenize(question))
    if not q_keywords:
        return FALLBACK_RESPONSE

    n = len(chunks)
    chunk_tokens = {c.chunk_id: set(tokenize(c.text)) for c in chunks}

    df = {}
    for kw in q_keywords:
        df[kw] = sum(1 for toks in chunk_tokens.values() if kw in toks)

    idf = {kw: math.log((n + 1) / (df[kw] + 1)) + 1 for kw in q_keywords}
    total_idf = sum(idf.values())

    scores = {}
    for c in chunks:
        toks = chunk_tokens[c.chunk_id]
        scores[c.chunk_id] = sum(idf[kw] for kw in q_keywords if kw in toks)

    if total_idf == 0 or max(scores.values()) == 0:
        return FALLBACK_RESPONSE

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1]
    best_ratio = top_score / total_idf

    ANSWER_THRESHOLD = 0.35
    if best_ratio < ANSWER_THRESHOLD:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(min(best_ratio * 0.8, 0.3), 2),
            "answerable": False
        }

    MULTI_CHUNK_RATIO = 0.6
    selected = [cid for cid, sc in ranked if sc >= top_score * MULTI_CHUNK_RATIO and sc > 0][:3]

    chunk_map = {c.chunk_id: c.text for c in chunks}
    top_chunk_id = ranked[0][0]
    top_text = chunk_map[top_chunk_id]

    sentences = split_sentences(top_text)
    if not sentences:
        answer = top_text.strip()
    else:
        sent_scores = []
        for s in sentences:
            s_toks = set(tokenize(s))
            sc = sum(idf[kw] for kw in q_keywords if kw in s_toks)
            sent_scores.append((sc, s))
        sent_scores.sort(key=lambda x: x[0], reverse=True)
        answer = sent_scores[0][1] if sent_scores[0][0] > 0 else top_text.strip()

    confidence = round(min(0.95, 0.45 + best_ratio * 0.5), 2)

    return {
        "answer": answer,
        "citations": selected,
        "confidence": confidence,
        "answerable": True
    }
