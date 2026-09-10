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
import json
from pypdf import PdfReader
import io
from docx import Document
from ddgs import DDGS

load_dotenv()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "super-secret-key-change-this"))
templates = Jinja2Templates(directory="templates")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

USERS_FILE = "users.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def history_file(email):
    return f"history_{email.replace('@', '_at_').replace('.', '_dot_')}.json"


def reminders_file(email):
    return f"reminders_{email.replace('@', '_at_').replace('.', '_dot_')}.json"


def load_history(email):
    path = history_file(email)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return [
            {"role": "system", "content": "You are Benganson AI, a helpful personal assistant. You are not ChatGPT and you should never refer to yourself as ChatGPT."}
        ]


def save_history(email, history):
    with open(history_file(email), "w") as f:
        json.dump(history, f)


def load_reminders(email):
    path = reminders_file(email)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return []


def save_reminders(email, reminders):
    with open(reminders_file(email), "w") as f:
        json.dump(reminders, f)


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return f"{salt}${hashed}"


def verify_password(password, stored):
    salt, hashed = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return hmac.compare_digest(check, hashed)


users = load_users()


def require_login(request: Request):
    if not request.session.get("user"):
        raise HTTPException(status_code=401, detail="Not logged in")
    return True


AUTH_PAGE_STYLE = """
<style>
    body { background:#0a0a14; color:#f5f4fa; font-family:sans-serif; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
    .card { background:#11101b; padding:30px; border-radius:12px; text-align:center; width:280px; border:1px solid rgba(255,255,255,0.08); }
    h2 { background:linear-gradient(135deg,#6366f1,#a855f7); -webkit-background-clip:text; background-clip:text; color:transparent; margin-top:0; }
    input { padding:10px; border-radius:8px; border:1px solid #2a2840; margin-bottom:12px; width:100%; background:#171625; color:#fff; box-sizing:border-box; }
    button { padding:10px 20px; border-radius:8px; border:none; background:linear-gradient(135deg,#6366f1,#a855f7); color:#fff; font-weight:bold; cursor:pointer; width:100%; }
    a { color:#a855f7; font-size:13px; }
    p.error { color:#ec4899; font-size:13px; }
</style>
"""


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    error_html = f"<p class='error'>{error}</p>" if error else ""
    return f"""
    <html><body>
    {AUTH_PAGE_STYLE}
    <div class="card">
        <h2>Benganson AI</h2>
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
    user = users.get(email)

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
        <h2>Create account</h2>
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

    if email in users:
        return RedirectResponse(url="/signup?error=Email+already+registered", status_code=303)

    if len(password) < 6:
        return RedirectResponse(url="/signup?error=Password+too+short", status_code=303)

    users[email] = {"password": hash_password(password)}
    save_users(users)

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
    conversation_history = [
        {"role": "system", "content": "You are Benganson AI, a helpful personal assistant. You are not ChatGPT and you should never refer to yourself as ChatGPT."}
    ]
    save_history(email, conversation_history)
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