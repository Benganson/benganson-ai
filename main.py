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
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "super-secret-key-change-this")
)
templates = Jinja2Templates(directory="templates")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["bengansonai"]
users_col = db["users"]
history_col = db["history"]
reminders_col = db["reminders"]
memory_col = db["memory"]

DEFAULT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are Benganson AI, a helpful personal assistant. "
        "You are not ChatGPT and you should never refer to yourself as ChatGPT. "
        "You were created by Abraham Benjamin Abraham Gandi, a Mechatronics Engineering "
        "student at Ahmadu Bello University, Zaria, Kaduna State, Nigeria, originally from "
        "Bauchi State, Nigeria. If asked who created or built you, or about your creator, "
        "share these facts naturally. Do not invent any other personal details about your "
        "creator beyond what is stated here — if asked something you don't actually know "
        "(his age, exact plans, other projects, etc.), say you don't have that information. "
        "If you are unsure about something, don't have up-to-date information (such as "
        "current events, recent news, live scores, prices, or someone's current role), or "
        "simply don't know the answer, say so honestly instead of guessing — this helps "
        "make sure accurate information can be looked up when needed."
    )
}

# Phrases that suggest the AI doesn't actually know the answer and could
# use a web search to find out. Checked in lowercase against the reply.
UNCERTAINTY_MARKERS = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i don't have access to real-time",
    "i do not have access to real-time",
    "i don't have real-time",
    "i do not have real-time",
    "as of my last update",
    "as of my knowledge cutoff",
    "as of my last training",
    "i can't browse the internet",
    "i cannot browse the internet",
    "i'm unable to browse",
    "i am unable to browse",
    "i don't have information on",
    "i do not have information on",
    "i don't have current information",
    "i do not have current information",
    "i don't have the current",
    "i do not have the current",
    "beyond my knowledge",
    "i don't have the ability to access",
    "i do not have the ability to access",
    "i don't have that information",
    "i do not have that information",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_history(email):
    doc = history_col.find_one({"email": email})
    if doc:
        return doc["messages"]
    return [DEFAULT_SYSTEM_MESSAGE]


def save_history(email, messages):
    history_col.update_one(
        {"email": email},
        {"$set": {"messages": messages}},
        upsert=True
    )


def load_memory(email):
    doc = memory_col.find_one({"email": email})
    if doc:
        return doc["facts"]
    return []


def save_memory(email, facts):
    memory_col.update_one(
        {"email": email},
        {"$set": {"facts": facts}},
        upsert=True
    )


def build_system_message(email):
    """
    Builds a fresh system message every time, combining the base
    Benganson AI identity with any personal facts remembered about
    this specific user. This way, remembered facts stay up to date
    even after a conversation is cleared.
    """

    facts = load_memory(email)

    content = DEFAULT_SYSTEM_MESSAGE["content"]

    if facts:
        facts_text = "\n".join(f"- {fact}" for fact in facts)
        content += (
            "\n\nHere are some facts you have been told to remember about "
            "this specific user from previous conversations. Use them "
            "naturally when relevant, but don't force them into every "
            "reply, and don't repeat them back unless it makes sense to:\n"
            f"{facts_text}"
        )

    return {
        "role": "system",
        "content": content
    }


def build_groq_messages(conversation_history, email):
    """
    Keep the full conversation history in MongoDB,
    but only send recent messages to Groq to avoid
    exceeding the Groq token-per-minute limit.

    The system message is always rebuilt fresh (instead of reusing
    whatever was saved in history) so remembered facts about the
    user are always current.
    """

    non_system_messages = [
        message
        for message in conversation_history
        if message.get("role") != "system"
    ]

    recent_messages = non_system_messages[-10:]

    system_message = build_system_message(email)

    messages_for_groq = [system_message] + recent_messages

    return [
        {
            key: value
            for key, value in message.items()
            if key != "time"
        }
        for message in messages_for_groq
    ]


def reply_shows_uncertainty(reply):
    """
    Checks whether the AI's reply suggests it doesn't actually know
    the answer, so we know when to automatically fall back to a web
    search instead of making the user turn Web Search on manually.
    """

    lowered = reply.lower()

    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


def search_the_web(query):
    """
    Runs a DuckDuckGo search and returns a plain-text summary of the
    top results, or a fallback message if the search fails.
    """

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

        return "\n\n".join(
            [
                f"{r['title']}: {r['body']} "
                f"(Source: {r['href']})"
                for r in results
            ]
        )

    except Exception as e:
        print("Search error:", e)
        return "No results found."


def get_ai_reply(conversation_history, email, original_message):
    """
    Gets a reply from Groq based on the current conversation history.

    If the reply suggests the AI doesn't actually know the answer
    (see reply_shows_uncertainty), it automatically searches the web
    for the original message and asks again using those results —
    so the user doesn't need to manually turn on Web Search for
    every question the AI might not already know.
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=build_groq_messages(conversation_history, email)
        )

        reply = response.choices[0].message.content

    except Exception as e:
        print("Groq error:", e)
        return (
            "Sorry, I'm having trouble connecting right now. "
            "Please try again in a moment."
        )

    if reply_shows_uncertainty(reply):
        summary = search_the_web(original_message)

        search_history = conversation_history + [
            {
                "role": "assistant",
                "content": reply
            },
            {
                "role": "user",
                "content": (
                    f"Search the web for: {original_message}\n\n"
                    f"Here are the top results:\n\n{summary}\n\n"
                    "Using these results, give me an accurate, "
                    "up-to-date answer, and mention the source(s)."
                )
            }
        ]

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=build_groq_messages(search_history, email)
            )

            reply = response.choices[0].message.content

        except Exception as e:
            print("Groq error:", e)
            # keep the original (uncertain) reply if this follow-up call fails

    return reply


def load_reminders(email):
    doc = reminders_col.find_one({"email": email})
    if doc:
        return doc["items"]
    return []


def save_reminders(email, items):
    reminders_col.update_one(
        {"email": email},
        {"$set": {"items": items}},
        upsert=True
    )


def get_user(email):
    return users_col.find_one({"email": email})


def create_user(email, password_hash):
    users_col.insert_one({
        "email": email,
        "password": password_hash
    })


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    ).hex()

    return f"{salt}${hashed}"


def verify_password(password, stored):
    salt, hashed = stored.split("$")

    check = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    ).hex()

    return hmac.compare_digest(check, hashed)


def require_login(request: Request):
    if not request.session.get("user"):
        raise HTTPException(
            status_code=401,
            detail="Not logged in"
        )
    return True


# ============================================================
# AUTH PAGE STYLING
# ============================================================

AUTH_PAGE_STYLE = """
<meta name="viewport" content="width=device-width, initial-scale=1">

<link rel="preconnect" href="https://fonts.googleapis.com">

<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    * {
        box-sizing: border-box;
    }

    body {
        background: #0a0a14;
        color: #f5f4fa;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        margin: 0;
        padding: 20px;
    }

    .card {
        background: #11101b;
        padding: 32px 28px;
        border-radius: 16px;
        text-align: center;
        width: 100%;
        max-width: 300px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 18px;
    }

    .brand h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 700;
        background: linear-gradient(
            135deg,
            #60a5fa,
            #a855f7,
            #ec4899
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
    }


    /* ========================================================
       BENGANSON AI LOGIN BRANDING
       Matches the main app branding
       ======================================================== */

    .login-brand {
        flex-direction: column;
        gap: 10px;
        margin-bottom: 26px;
    }

    .login-brand svg {
        width: 64px;
        height: 64px;
        flex-shrink: 0;
        filter: drop-shadow(
            0 0 16px rgba(168, 85, 247, 0.35)
        );
    }

    .login-brand h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }


    input {
        padding: 11px;
        border-radius: 8px;
        border: 1px solid #2a2840;
        margin-bottom: 12px;
        width: 100%;
        background: #171625;
        color: #fff;
        font-size: 15px;
    }

    button {
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        background: linear-gradient(
            135deg,
            #6366f1,
            #a855f7
        );
        color: #fff;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        font-size: 15px;
    }

    a {
        color: #a855f7;
        font-size: 13px;
    }

    p.error {
        color: #ec4899;
        font-size: 13px;
    }
</style>
"""


# ============================================================
# BENGANSON AI LOGO
# Same logo shape and gradient as the main app
# ============================================================

AUTH_LOGO_SVG = """
<svg class="auth-logo" viewBox="0 0 24 24" aria-hidden="true">
    <defs>
        <linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#60a5fa"/>
            <stop offset="50%" stop-color="#a855f7"/>
            <stop offset="100%" stop-color="#ec4899"/>
        </linearGradient>
    </defs>

    <path
        d="M12 2C12.8 7 17 11.2 22 12C17 12.8 12.8 17 12 22C11.2 17 7 12.8 2 12C7 11.2 11.2 7 12 2Z"
        fill="url(#logoGrad)"
    />
</svg>
"""


# ============================================================
# LOGIN
# ============================================================

@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    error_html = f"<p class='error'>{error}</p>" if error else ""

    return f"""
    <html>
    <body>

    {AUTH_PAGE_STYLE}

    <div class="card">

        <div class="brand login-brand">
            {AUTH_LOGO_SVG}
            <h2>Benganson AI</h2>
        </div>

        <form method="post" action="/login">

            <input
                type="email"
                name="email"
                placeholder="Email"
                required
            >

            <input
                type="password"
                name="password"
                placeholder="Password"
                required
            >

            {error_html}

            <button type="submit">
                Log In
            </button>

        </form>

        <p>
            No account?
            <a href="/signup">Sign up</a>
        </p>

    </div>

    </body>
    </html>
    """


@app.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.lower().strip()
    user = get_user(email)

    if not user or not verify_password(password, user["password"]):
        return RedirectResponse(
            url="/login?error=Invalid+email+or+password",
            status_code=303
        )

    request.session["user"] = email

    return RedirectResponse(
        url="/",
        status_code=303
    )


# ============================================================
# SIGN UP
# ============================================================

@app.get("/signup", response_class=HTMLResponse)
def signup_page(error: str = ""):
    error_html = f"<p class='error'>{error}</p>" if error else ""

    return f"""
    <html>
    <body>

    {AUTH_PAGE_STYLE}

    <div class="card">

        <div class="brand">
            {AUTH_LOGO_SVG}
            <h2>Create account</h2>
        </div>

        <form method="post" action="/signup">

            <input
                type="email"
                name="email"
                placeholder="Email"
                required
            >

            <input
                type="password"
                name="password"
                placeholder="Password (min 6 chars)"
                required
            >

            {error_html}

            <button type="submit">
                Sign Up
            </button>

        </form>

        <p>
            Already have an account?
            <a href="/login">Log in</a>
        </p>

    </div>

    </body>
    </html>
    """


@app.post("/signup")
def signup_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.lower().strip()

    if get_user(email):
        return RedirectResponse(
            url="/signup?error=Email+already+registered",
            status_code=303
        )

    if len(password) < 6:
        return RedirectResponse(
            url="/signup?error=Password+too+short",
            status_code=303
        )

    create_user(
        email,
        hash_password(password)
    )

    request.session["user"] = email

    return RedirectResponse(
        url="/",
        status_code=303
    )


# ============================================================
# LOGOUT
# ============================================================

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")


# ============================================================
# MAIN APP
# ============================================================

@app.get("/")
def read_root(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user_email": request.session.get("user")
        }
    )


# ============================================================
# FILE UPLOAD
# ============================================================

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(require_login)
):
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

        text = "\n".join(
            [para.text for para in doc.paragraphs]
        )

    else:
        text = contents.decode("utf-8")

    conversation_history.append({
        "role": "user",
        "content": (
            f"I've uploaded a file named '{file.filename}'. "
            f"Here is its content:\n\n{text}\n\n"
            "Please use this content to answer any questions "
            "I ask about it."
        ),
        "time": now_iso()
    })

    save_history(
        email,
        conversation_history
    )

    return {
        "status": (
            f"File '{file.filename}' uploaded "
            "and added to memory."
        )
    }


# ============================================================
# CLEAR FILES
# ============================================================

@app.get("/clear-files")
def clear_files(
    request: Request,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    conversation_history = load_history(email)

    conversation_history = [
        msg
        for msg in conversation_history
        if not msg["content"].startswith(
            "I've uploaded a file named"
        )
    ]

    save_history(
        email,
        conversation_history
    )

    return {
        "status": "Uploaded file content cleared."
    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/history")
def get_history(
    request: Request,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    return {
        "history": load_history(email)
    }


# ============================================================
# RESET MEMORY (conversation history only — not remembered facts)
# ============================================================

@app.get("/reset")
def reset(
    request: Request,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    save_history(
        email,
        [DEFAULT_SYSTEM_MESSAGE]
    )

    return {
        "status": "Memory cleared"
    }


# ============================================================
# ADD REMINDER
# ============================================================

@app.post("/add-reminder")
def add_reminder(
    request: Request,
    text: str,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    reminders = load_reminders(email)

    reminders.append({
        "text": text,
        "done": False,
        "time": now_iso()
    })

    save_reminders(
        email,
        reminders
    )

    return {
        "status": "Reminder added.",
        "reminders": reminders
    }


# ============================================================
# GET REMINDERS
# ============================================================

@app.get("/reminders")
def get_reminders(
    request: Request,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    return {
        "reminders": load_reminders(email)
    }


# ============================================================
# DELETE REMINDER
# ============================================================

@app.get("/delete-reminder")
def delete_reminder(
    request: Request,
    index: int,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    reminders = load_reminders(email)

    if 0 <= index < len(reminders):
        reminders.pop(index)

        save_reminders(
            email,
            reminders
        )

    return {
        "reminders": reminders
    }


# ============================================================
# PERSONAL MEMORY (facts remembered about this specific user)
# ============================================================

@app.post("/add-memory")
def add_memory(
    request: Request,
    text: str,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    facts = load_memory(email)

    facts.append(text)

    save_memory(
        email,
        facts
    )

    return {
        "status": "Got it, I'll remember that.",
        "facts": facts
    }


@app.get("/memory")
def get_memory(
    request: Request,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    return {
        "facts": load_memory(email)
    }


@app.get("/delete-memory")
def delete_memory(
    request: Request,
    index: int,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    facts = load_memory(email)

    if 0 <= index < len(facts):
        facts.pop(index)

        save_memory(
            email,
            facts
        )

    return {
        "facts": facts
    }


# ============================================================
# WEB SEARCH (manual — triggered when the user turns Web Search on)
# ============================================================

@app.get("/web-search")
def web_search(
    request: Request,
    query: str,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    conversation_history = load_history(email)

    summary = search_the_web(query)

    conversation_history.append({
        "role": "user",
        "content": (
            f"Search the web for: {query}\n\n"
            f"Here are the top results:\n\n{summary}\n\n"
            "Using these results, give me an accurate, "
            "up-to-date answer, and mention the source(s)."
        ),
        "time": now_iso()
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=build_groq_messages(
                conversation_history,
                email
            )
        )

        reply = response.choices[0].message.content

    except Exception as e:
        reply = (
            "Sorry, I'm having trouble connecting right now. "
            "Please try again in a moment."
        )

        print("Groq error:", e)

    conversation_history.append({
        "role": "assistant",
        "content": reply,
        "time": now_iso()
    })

    save_history(
        email,
        conversation_history
    )

    return {
        "query": query,
        "reply": reply
    }


# ============================================================
# CHAT (automatically falls back to a web search if the AI
# doesn't actually know the answer — see get_ai_reply)
# ============================================================

@app.get("/chat")
def chat(
    request: Request,
    message: str,
    _: bool = Depends(require_login)
):
    email = request.session.get("user")

    conversation_history = load_history(email)

    conversation_history.append({
        "role": "user",
        "content": message,
        "time": now_iso()
    })

    reply = get_ai_reply(conversation_history, email, message)

    conversation_history.append({
        "role": "assistant",
        "content": reply,
        "time": now_iso()
    })

    save_history(
        email,
        conversation_history
    )

    return {
        "user_message": message,
        "reply": reply
    }