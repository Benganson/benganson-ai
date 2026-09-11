from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import hashlib
import hmac
import secrets
from groq import Groq
from pymongo import MongoClient
from pypdf import PdfReader
import io
from docx import Document
from ddgs import DDGS

load_dotenv()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "super-secret-key-change-this"))
templates = Jinja2Templates(directory="templates")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["bengansonai"]
users_col = db["users"]
history_col = db["history"]
reminders_col = db["reminders"]

DEFAULT_SYSTEM_MESSAGE = {"role": "system", "content": "You are Benganson AI, a helpful personal assistant. You are not ChatGPT and you should never refer to yourself as ChatGPT."}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_history(email):
    doc = history_col.find_one({"email": email})
    if doc:
        return doc["messages"]
    return [DEFAULT_SYSTEM_MESSAGE]


def save_history(email, messages):
    history_col.update_one({"email": email}, {"$set": {"messages": messages}}, upsert=True)


def load_reminders(email):
    doc = reminders_col.find_one({"email": email})
    if doc:
        return doc["items"]
    return []


def save_reminders(email, items):
    reminders_col.update_one({"email": email}, {"$set": {"items": items}}, upsert=True)


def get_user(email):
    return users_col.find_one({"email": email})


def create_user(email, password_hash):
    users_col.insert_one({"email": email, "password": password_hash})


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return f"{salt}${hashed}"


def verify_password(password, stored):
    salt, hashed = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return hmac.compare_digest(check, hashed)


def require_login(request: Request):
    if not request.session.get("user"):
        raise HTTPException(status_code=401, detail="Not logged in")
    return True


AUTH_PAGE_STYLE = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    * { box-sizing: border-box; }
    body { background:#0a0a14; color:#f5f4fa; font-family:'Inter',sans-serif; display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; padding:20px; }
    .card { background:#11101b; padding:32px 28px; border-radius:16px; text-align:center; width:100%; max-width:300px; border:1px solid rgba(255,255,255,0.08); }
    .brand { display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:18px; }
    .brand h2 { font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; background:linear-gradient(135deg,#60a5fa,#a855f7,#ec4899); -webkit-background-clip:text; background-clip:text; color:transparent; margin:0; }

    /* Benganson AI login branding — matches the main app */
    .login-brand {
        flex-direction:column;
        gap:10px;
        margin-bottom:26px;
    }
    .login-brand svg {
        width:64px;
        height:64px;
        filter:drop-shadow(0 0 16px rgba(168,85,247,0.35));
    }
    .login-brand h2 {
        font-size:28px;
        font-weight:800;
        letter-spacing:-0.8px;
    }
    input { padding:11px; border-radius:8px; border:1px solid #2a2840; margin-bottom:12px; width:100%; background:#171625; color:#fff; font-size:15px; }
    button { padding:12px 20px; border-radius:8px; border:none; background:linear-gradient(135deg,#6366f1,#a855f7); color:#fff; font-weight:bold; cursor:pointer; width:100%; font-size:15px; }
    a { color:#a855f7; font-size:13px; }
    p.error { color:#ec4899; font-size:13px; }
</style>
"""

AUTH_LOGO_SVG = """
<svg width="30" height="30" viewBox="0 0 24 24">
    <defs>
        <linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#60a5fa"/>
            <stop offset="50%" stop-color="#a855f7"/>
            <stop offset="100%" stop-color="#ec4899"/>
        </linearGradient>
    </defs>
    <path d="M12 2C12.8 7 17 11.2 22 12C17 12.8 12.8 17 12 22C11.2 17 7 12.8 2 12C7 11.2 11.2 7 12 2Z" fill="url(#logoGrad)"/>
</svg>
"""


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    error_html = f"<p class='error'>{error}</p>" if error else ""
    return f"""
    <html><body>
    {AUTH_PAGE_STYLE}
    <div class="card">
        <div class="brand login-brand">{AUTH_LOGO_SVG}<h2>Benganson AI</h2></div>
        <form method="post" action="/login">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            {error_html}
            <button type="submit">Log In</button>
        </form>
        <p>No account? <a href="/signup">Sign up</a></p>
    </div>
    </body></html>
    """


@app.post("/login")
def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()
    user = get_user(email)

    if not user or not verify_password(password, user["password"]):
        return RedirectResponse(url="/login?error=Invalid+email+or+password", status_code=303)

    request.session["user"] = email
    return RedirectResponse(url="/", status_code=303)


@app.get("/signup", response_class=HTMLResponse)
def signup_page(error: str = ""):
    error_html = f"<p class='error'>{error}</p>" if error else ""
    return f"""
    <html><body>
    {AUTH_PAGE_STYLE}
    <div class="card">
        <div class="brand">{AUTH_LOGO_SVG}<h2>Create account</h2></div>
        <form method="post" action="/signup">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password (min 6 chars)" required>
            {error_html}
            <button type="submit">Sign Up</button>
        </form>
        <p>Already have an account? <a href="/login">Log in</a></p>
    </div>
    </body></html>
    """


@app.post("/signup")
def signup_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()

    if get_user(email):
        return RedirectResponse(url="/signup?error=Email+already+registered", status_code=303)

    if len(password) < 6:
        return RedirectResponse(url="/signup?error=Password+too+short", status_code=303)

    create_user(email, hash_password(password))

    request.session["user"] = email
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")


@app.get("/")
def read_root(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(request, "index.html", {"user_email": request.session.get("user")})


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), _: bool = Depends(require_login)):
    email = request.session.get("user")
    conversation_history = load_history(email)
    contents = await file.read()

    if file.filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.filename.endswith(".docx"):
        doc = Document(io.BytesIO(contents))
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = contents.decode("utf-8")

    conversation_history.append({
        "role": "user",
        "content": f"I've uploaded a file named '{file.filename}'. Here is its content:\n\n{text}\n\nPlease use this content to answer any questions I ask about it.",
        "time": now_iso()
    })
    save_history(email, conversation_history)

    return {"status": f"File '{file.filename}' uploaded and added to memory."}


@app.get("/clear-files")
def clear_files(request: Request, _: bool = Depends(require_login)):
    email = request.session.get("user")
    conversation_history = load_history(email)
    conversation_history = [
        msg for msg in conversation_history
        if not msg["content"].startswith("I've uploaded a file named")
    ]
    save_history(email, conversation_history)
    return {"status": "Uploaded file content cleared."}


@app.get("/history")
def get_history(request: Request, _: bool = Depends(require_login)):
    email = request.session.get("user")
    return {"history": load_history(email)}


@app.get("/reset")
def reset(request: Request, _: bool = Depends(require_login)):
    email = request.session.get("user")
    save_history(email, [DEFAULT_SYSTEM_MESSAGE])
    return {"status": "Memory cleared"}


@app.post("/add-reminder")
def add_reminder(request: Request, text: str, _: bool = Depends(require_login)):
    email = request.session.get("user")
    reminders = load_reminders(email)
    reminders.append({"text": text, "done": False, "time": now_iso()})
    save_reminders(email, reminders)
    return {"status": "Reminder added.", "reminders": reminders}


@app.get("/reminders")
def get_reminders(request: Request, _: bool = Depends(require_login)):
    email = request.session.get("user")
    return {"reminders": load_reminders(email)}


@app.get("/delete-reminder")
def delete_reminder(request: Request, index: int, _: bool = Depends(require_login)):
    email = request.session.get("user")
    reminders = load_reminders(email)
    if 0 <= index < len(reminders):
        reminders.pop(index)
        save_reminders(email, reminders)
    return {"reminders": reminders}


@app.get("/web-search")
def web_search(request: Request, query: str, _: bool = Depends(require_login)):
    email = request.session.get("user")
    conversation_history = load_history(email)

    try:
        results = DDGS().text(query, max_results=5)
        summary = "\n\n".join([f"{r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        summary = "No results found."
        print("Search error:", e)

    conversation_history.append({
        "role": "user",
        "content": f"Search the web for: {query}\n\nHere are the top results:\n\n{summary}\n\nUsing these results, give me an accurate, up-to-date answer, and mention the source(s).",
        "time": now_iso()
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{k: v for k, v in m.items() if k != "time"} for m in conversation_history]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, I'm having trouble connecting right now. Please try again in a moment."
        print("Groq error:", e)

    conversation_history.append({"role": "assistant", "content": reply, "time": now_iso()})
    save_history(email, conversation_history)

    return {"query": query, "reply": reply}


@app.get("/chat")
def chat(request: Request, message: str, _: bool = Depends(require_login)):
    email = request.session.get("user")
    conversation_history = load_history(email)

    conversation_history.append({"role": "user", "content": message, "time": now_iso()})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{k: v for k, v in m.items() if k != "time"} for m in conversation_history]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, I'm having trouble connecting right now. Please try again in a moment."
        print("Groq error:", e)

    conversation_history.append({"role": "assistant", "content": reply, "time": now_iso()})
    save_history(email, conversation_history)

    return {
        "user_message": message,
        "reply": reply
    }