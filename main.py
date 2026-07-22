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

def extract_boost_words(question: str):
    raw_words = re.findall(r"[A-Za-z0-9']+", question)
    boost = set()
    for i, w in enumerate(raw_words):
        if i == 0:
            continue
        lw = w.lower()
        if lw in STOPWORDS or len(w) <= 1:
            continue
        if w.isupper() or (w[0].isupper() and not w.islower()):
            boost.add(lw)
    return boost

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

    boost_words = extract_boost_words(question) & q_keywords

    n = len(chunks)
    chunk_tokens = {c.chunk_id: set(tokenize(c.text)) for c in chunks}

    df = {kw: sum(1 for toks in chunk_tokens.values() if kw in toks) for kw in q_keywords}

    present_keywords = [kw for kw in q_keywords if df[kw] > 0]
    if not present_keywords:
        return FALLBACK_RESPONSE

    # Guard 1: a named entity in the question must actually appear somewhere
    # in the chunks -- an incidental shared generic word isn't enough.
    if boost_words and not any(bw in present_keywords for bw in boost_words):
        return FALLBACK_RESPONSE

    # Guard 2: naming the right entity isn't sufficient on its own -- the
    # chunks must also cover a reasonable share of the OTHER terms in the
    # question (the actual attribute/fact being asked about), otherwise the
    # entity might be mentioned without the requested detail being present.
    coverage = len(present_keywords) / len(q_keywords)
    if coverage < 0.4:
        return FALLBACK_RESPONSE

    def weight(kw):
        idf = math.log((n + 1) / (df[kw] + 1)) + 1
        return idf * (1.8 if kw in boost_words else 1.0)

    idf = {kw: weight(kw) for kw in present_keywords}
    total_idf = sum(idf.values())

    scores = {}
    for c in chunks:
        toks = chunk_tokens[c.chunk_id]
        scores[c.chunk_id] = sum(idf[kw] for kw in present_keywords if kw in toks)

    if total_idf == 0 or max(scores.values()) == 0:
        return FALLBACK_RESPONSE

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1]
    best_ratio = top_score / total_idf

    ANSWER_THRESHOLD = 0.4
    if best_ratio < ANSWER_THRESHOLD:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(min(best_ratio * 0.8, 0.3), 2),
            "answerable": False
        }

    top_chunk_id = ranked[0][0]

    if boost_words and not any(bw in chunk_tokens[top_chunk_id] for bw in boost_words):
        return FALLBACK_RESPONSE

    MULTI_CHUNK_RATIO = 0.6
    selected = [cid for cid, sc in ranked if sc >= top_score * MULTI_CHUNK_RATIO and sc > 0][:3]

    chunk_map = {c.chunk_id: c.text for c in chunks}
    answer_parts = [chunk_map[cid].strip() for cid in selected]
    answer = " ".join(answer_parts)

    confidence = round(min(0.95, 0.45 + best_ratio * 0.5), 2)

    return {
        "answer": answer,
        "citations": selected,
        "confidence": confidence,
        "answerable": True
    }
