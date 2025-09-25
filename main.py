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

REASONING_MODELS = {"o3", "o3-pro", "gpt-5-thinking"}

@app.post("/reason")
def reason(body: dict):
    expected = os.environ.get("PROXY_TOKEN", "secret")
    if body.get("token") != expected:
        return {"ok": False, "where": "auth", "error": "Bad token in body (token field)"}

    prompt = body.get("prompt")
    model = body.get("model")
    temperature = body.get("temperature")  # opzionale per modelli non-reasoning

    if model not in REASONING_MODELS:
        # Modelli “non-reasoning” (es. 4o) – puoi tenere temperature
        payload = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
    else:
        # Modelli reasoning: NIENTE temperature
        payload = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            # opzionale: suggerisci lo sforzo di reasoning
            "reasoning": {"effort": "medium"},
            # opzionale: limita l’output
            "max_output_tokens": 2048
        }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        json=payload,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=40
    )
    if r.status_code != 200:
        return {"ok": False, "where": "openai", "status": r.status_code, "error": r.text[:500]}

    try:
        data = r.json()
    except Exception:
        return {"ok": False, "where": "openai", "error": f"Invalid JSON: {r.text[:300]}"}

    text = data.get("output_text")
    if not text:
        try:
            text = data["output"][0]["content"][0]["text"]
        except Exception:
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
