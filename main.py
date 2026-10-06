import io, json, os, re, uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sarvamai import SarvamAI

DATA = Path("data"); DATA.mkdir(exist_ok=True)
DB, CFG = DATA / "chunks.json", DATA / "settings.json"
ADMIN = os.getenv("ADMIN_PASSWORD", "change-me")
THINK = os.getenv("THINK") or None  # low | medium | high; empty = fastest
DEFAULT = {
    "name": "Hazel",
    "model": os.getenv("MODEL", "sarvam-105b"),
    "welcome": "Namaste 👋",
    "chips": "Stress kam karne ke 3 tips batao\nEk psychology concept aasan bhasha mein samjhao\nMeri aaj ki study plan banao",
    "bg": "", "bgimg": "", "accent": "", "voice": "", "rate": 1, "pitch": 1,
    "temp": 0.7, "maxt": 2000, "think": "", "books": 1, "wiki": 0,
    "provider": "sarvam", "backup_provider": "sarvam", "backup_model": "",
    "dost_name": "Dost", "nick": "", "greeting": "Bol, kya chal raha hai?", "humor": 5, "formality": 3,
    "length": "medium", "emotion": "warm", "lang": "Hinglish", "volume": 1, "font": "",
    "f_mic": 1, "f_copy": 1, "f_listen": 1, "f_like": 1, "f_regen": 1, "f_edit": 1, "f_share": 1,
    "m_voice": 1, "m_dost": 1, "m_projects": 1, "m_web": 1, "m_images": 1, "m_files": 1, "m_code": 1, "m_memory": 1, "m_tasks": 1, "m_help": 1,
    "prompt": "You are Hazel, a warm, sharp AI assistant. Reply in the user's language "
              "(Hindi, Hinglish or English). Be concise unless asked for depth.",
}
client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
app = FastAPI()


def load(p, d):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else d

def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")

def auth(x):
    if x != ADMIN:
        raise HTTPException(401, "wrong admin password")

def words(t):  # splits on spaces/punctuation, so Hindi matras stay intact
    return set(re.findall(r"[^\s.,;:!?()\[\]\"'।]+", t.lower()))

def chunk(text, size=900):
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > size and cur:
            out.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out

def search(q, k=4):
    qs, hits = words(q), []
    for c in load(DB, []):
        s = len(qs & words(c["text"]))
        if s:
            hits.append((s, c))
    return [c for _, c in sorted(hits, key=lambda h: -h[0])[:k]]


def wiki(q):  # Wikipedia connector, no key needed
    try:
        import httpx
        lang = "hi" if re.search("[\u0900-\u097F]", q) else "en"
        r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php", timeout=6, headers={"User-Agent": "HazelAI/1.0"},
                      params={"action": "query", "list": "search", "srsearch": q, "srlimit": 3, "format": "json"})
        return "Wikipedia:\n" + "\n".join(re.sub("<[^>]+>", "", h["title"] + ": " + h["snippet"]) for h in r.json()["query"]["search"])
    except Exception:
        return ""


class Chat(BaseModel):
    messages: list[dict]
    extra: str = ""


OAI = {"gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
       "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
       "grok": ("https://api.x.ai/v1", "XAI_API_KEY")}


def persona(c):
    nick = f" (nickname: {c['nick']})" if c["nick"] else ""
    return (c["prompt"] + f"\nYour name is {c['dost_name']}{nick}. You are an AI and never claim to be human."
            f" Humor {c['humor']}/10, formality {c['formality']}/10, reply length {c['length']}, emotional style {c['emotion']}, main language {c['lang']}."
            " Prefer plain conversational sentences; use markdown only for real lists or code.")


def run(provider, model, system, msgs, cfg):  # one branch per AI provider; keys come from env vars only
    msgs = [{"role": "system", "content": system}] + msgs
    if provider == "sarvam":
        if not model.startswith("sarvam"):
            model = DEFAULT["model"]
        for ch in client.chat.completions(model=model, messages=msgs, max_tokens=int(cfg["maxt"]),
                                          temperature=float(cfg["temp"]), reasoning_effort=cfg["think"] or THINK, stream=True):
            if ch.choices and ch.choices[0].delta.content:
                yield ch.choices[0].delta.content
    else:
        import httpx
        base, env = OAI[provider]
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set")
        body = {"model": model, "messages": msgs, "stream": True,
                "temperature": float(cfg["temp"]), "max_tokens": int(cfg["maxt"])}
        with httpx.stream("POST", base + "/chat/completions", json=body, timeout=60,
                          headers={"Authorization": "Bearer " + key}) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"{provider} error {r.status_code}: {r.read().decode()[:200]}")
            for line in r.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    d = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if d:
                        yield d


@app.post("/api/chat")
def chat(req: Chat):
    cfg = {**DEFAULT, **load(CFG, {})}
    msgs = [{"role": m["role"], "content": m["content"]} for m in req.messages[-20:]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    q = msgs[-1]["content"]
    ctx = "\n---\n".join(f'[{c["src"]}] {c["text"]}' for c in search(q)) if cfg["books"] else ""
    if cfg["wiki"]:
        ctx += "\n" + wiki(q)
    system = persona(cfg) + (("\nProject instructions: " + req.extra) if req.extra else "") + f"\nNow: {datetime.now():%A %d %B %Y, %I:%M %p}."
    if ctx.strip():
        system += "\n\nUse this knowledge when relevant:\n" + ctx

    def gen():
        tries = [(cfg["provider"], cfg["model"])]
        if cfg["backup_model"]:
            tries.append((cfg["backup_provider"], cfg["backup_model"]))
        err = ""
        for prov, model in tries:
            got = False
            try:
                for t in run(prov, model, system, msgs, cfg):
                    got = True
                    yield t
                if got:
                    return
            except Exception as e:
                err = str(e)
                if got:
                    break
        yield f"\n[error: {err or 'no reply'}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/api/settings")
def public_settings():
    return {k: v for k, v in {**DEFAULT, **load(CFG, {})}.items() if k not in ("prompt", "model", "temp", "maxt", "think")}

@app.get("/api/admin/settings")
def get_cfg(x_admin: str = Header("")):
    auth(x_admin)
    return {**DEFAULT, **load(CFG, {})}

@app.post("/api/admin/settings")
def set_cfg(c: dict, x_admin: str = Header("")):
    auth(x_admin)
    save(CFG, {k: c[k] for k in DEFAULT if k in c})
    return {"ok": True}

@app.post("/api/admin/upload")
async def upload(f: UploadFile = File(...), x_admin: str = Header("")):
    auth(x_admin)
    raw = await f.read()
    if f.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8", "ignore")
    parts = chunk(text)
    save(DB, load(DB, []) + [{"id": uuid.uuid4().hex[:8], "src": f.filename, "text": c} for c in parts])
    return {"chunks": len(parts)}

@app.get("/api/admin/docs")
def docs(x_admin: str = Header("")):
    auth(x_admin)
    out = {}
    for c in load(DB, []):
        out[c["src"]] = out.get(c["src"], 0) + 1
    return out

@app.delete("/api/admin/docs/{name}")
def remove(name: str, x_admin: str = Header("")):
    auth(x_admin)
    save(DB, [c for c in load(DB, []) if c["src"] != name])
    return {"ok": True}


AV = DATA / "avatar.jpg"


@app.get("/api/avatar")
def avatar():
    p = AV if AV.exists() else Path("avatar.jpg")
    return FileResponse(p) if p.exists() else Response(status_code=404)

@app.post("/api/admin/avatar")
async def set_avatar(f: UploadFile = File(...), x_admin: str = Header("")):
    auth(x_admin)
    AV.write_bytes(await f.read())
    return {"ok": True}

@app.post("/api/admin/test")
def test_ai(c: dict, x_admin: str = Header("")):
    import time
    auth(x_admin)
    t0 = time.time()
    try:
        out = "".join(run(c.get("provider", "sarvam"), c.get("model") or DEFAULT["model"], "Reply in one short line.",
                          [{"role": "user", "content": c.get("text") or "Say hello"}], {**DEFAULT, **load(CFG, {}), "maxt": 150}))
        return {"ok": True, "ms": int((time.time() - t0) * 1000), "text": out[:300]}
    except Exception as e:
        return {"ok": False, "ms": int((time.time() - t0) * 1000), "text": str(e)[:300]}


@app.get("/")
def home():
    return FileResponse("index.html")
