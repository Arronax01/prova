from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os, requests

app = FastAPI(title="Reasoning Proxy (o3 / GPT-5 Thinking)")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/responses"
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "secret")

class ReasonReq(BaseModel):
    prompt: str
    model: str            # "o3", "o3-pro", "gpt-5-thinking"
    temperature: float | None = 0.2
    extra_context: dict | None = None

@app.post("/reason")
def reason(body: ReasonReq, authorization: str | None = Header(default=None)):
    # 1) controlli base
    if authorization != f"Bearer {PROXY_TOKEN}":
        return {"ok": False, "where": "auth", "error": "Unauthorized (Bearer token mismatch)"}
    if body.model not in ["o3", "o3-pro", "gpt-5-thinking"]:
        return {"ok": False, "where": "input", "error": f"Model not allowed: {body.model}"}
    if not OPENAI_API_KEY:
        return {"ok": False, "where": "env", "error": "Missing OPENAI_API_KEY"}

    # 2) chiama Responses API con timeout e gestione errori
    payload = {
        "model": body.model,
        "input": [{"role": "user", "content": body.prompt}],
        "temperature": body.temperature or 0.2
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        r = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=40)
    except requests.exceptions.Timeout:
        return {"ok": False, "where": "openai", "error": "Timeout calling OpenAI"}
    except Exception as e:
        return {"ok": False, "where": "openai", "error": f"Request error: {e.__class__.__name__}: {e}"}

    # 3) gestisci esiti non-200 con messaggio leggibile
    if r.status_code != 200:
        txt = r.text
        return {"ok": False, "where": "openai", "status": r.status_code, "error": txt[:500]}

    data = {}
    try:
        data = r.json()
    except Exception:
        return {"ok": False, "where": "openai", "error": f"Invalid JSON from OpenAI: {r.text[:300]}"}

    # 4) estrai testo in modo robusto (varianti schema)
    text = None
    if isinstance(data, dict):
        # nuovo campo comodo se presente
        if "output_text" in data and isinstance(data["output_text"], str):
            text = data["output_text"]
        # struttura tipo output -> content -> text
        if not text:
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                pass
    if not text:
        text = str(data)[:2000]  # fallback: ritorna raw (utile per debug)

    tokens = (data.get("usage", {}) or {}).get("total_tokens", 0)
    return {"ok": True, "text": text, "tokens_used": tokens, "model_used": body.model}

@app.get("/")
def root():
    return {"ok": True, "routes": ["/", "/health", "/diag", "/reason (POST)"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diag")
def diag():
    return {
        "ok": True,
        "env_openai": bool(os.environ.get("OPENAI_API_KEY")),
        "env_token": bool(os.environ.get("PROXY_TOKEN"))
    }
