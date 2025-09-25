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
def reason(body: dict):
    # === auth via body (niente header Bearer) ===
    expected = os.environ.get("PROXY_TOKEN", "secret")
    token = body.get("token")
    if not token or token != expected:
        return {"ok": False, "where": "auth", "error": "Bad token in body (token field)"}

    prompt = body.get("prompt")
    model = body.get("model")
    temperature = body.get("temperature", 0.2)

    if model not in ["o3", "o3-pro", "gpt-5-thinking"]:
        return {"ok": False, "where": "input", "error": f"Model not allowed: {model}"}

    if not os.environ.get("OPENAI_API_KEY"):
        return {"ok": False, "where": "env", "error": "Missing OPENAI_API_KEY"}

    # chiamata alla Responses API con gestione errori "parlante"
    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            json={
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "temperature": temperature
            },
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            timeout=40
        )
    except requests.exceptions.Timeout:
        return {"ok": False, "where": "openai", "error": "Timeout calling OpenAI"}

    if r.status_code != 200:
        return {"ok": False, "where": "openai", "status": r.status_code, "error": r.text[:500]}

    try:
        data = r.json()
    except Exception:
        return {"ok": False, "where": "openai", "error": f"Invalid JSON: {r.text[:300]}"}

    text = None
    if isinstance(data, dict):
        text = data.get("output_text")
        if not text:
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                pass
    if not text:
        text = str(data)[:2000]

    tokens = (data.get("usage", {}) or {}).get("total_tokens", 0)
    return {"ok": True, "text": text, "tokens_used": tokens, "model_used": model}

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
