from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os, requests

app = FastAPI(title="Reasoning Proxy (o3 / GPT-5 Thinking)")

# --- ROUTES BASE: DEVONO RISPONDERE ---
@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

# --- /reason: aggiunto dopo aver visto / e /health funzionare ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/responses"
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "secret")

class ReasonReq(BaseModel):
    prompt: str
    model: str           # "o3", "o3-pro", "gpt-5-thinking"
    temperature: float | None = 0.2
    extra_context: dict | None = None

@app.post("/reason")
def reason(body: ReasonReq, authorization: str | None = Header(default=None)):
    if authorization != f"Bearer {PROXY_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if body.model not in ["o3", "o3-pro", "gpt-5-thinking"]:
        raise HTTPException(status_code=400, detail="Model not allowed")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    payload = {
        "model": body.model,
        "input": [{"role": "user", "content": body.prompt}],
        "temperature": body.temperature
    }
    r = requests.post(
        OPENAI_URL, json=payload,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)

    data = r.json()
    try:
        text = data["output"][0]["content"][0]["text"]
    except Exception:
        text = str(data)
    tokens = (data.get("usage", {}) or {}).get("total_tokens", 0)
    return {"text": text, "tokens_used": tokens, "model_used": body.model}
