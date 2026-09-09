from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
from groq import Groq
import json
from pypdf import PdfReader
import io
from docx import Document
from ddgs import DDGS

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HISTORY_FILE = "conversation_history.json"
REMINDERS_FILE = "reminders.json"

def load_reminders():
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE, "r") as f:
            return json.load(f)
    else:
        return []

def save_reminders(reminders):
    with open(REMINDERS_FILE, "w") as f:
        json.dump(reminders, f)

reminders = load_reminders()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    else:
        return [
            {"role": "system", "content": "You are Benganson AI, a helpful personal assistant. You are not ChatGPT and you should never refer to yourself as ChatGPT."}
        ]

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

conversation_history = load_history()


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )

@app.get("/clear-files")
def clear_files():
    global conversation_history
    conversation_history = [
        msg for msg in conversation_history
        if not msg["content"].startswith("I've uploaded a file named")
    ]
    save_history(conversation_history)
    return {"status": "Uploaded file content cleared."}
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
        "content": f"I've uploaded a file named '{file.filename}'. Here is its content:\n\n{text}\n\nPlease use this content to answer any questions I ask about it."
    })
    save_history(conversation_history)

    return {"status": f"File '{file.filename}' uploaded and added to memory."}

@app.post("/add-reminder")
def add_reminder(text: str):
    reminders.append({"text": text, "done": False})
    save_reminders(reminders)
    return {"status": "Reminder added.", "reminders": reminders}


@app.get("/reminders")
def get_reminders():
    return {"reminders": reminders}
@app.get("/delete-reminder")
def delete_reminder(index: int):
    if 0 <= index < len(reminders):
        reminders.pop(index)
        save_reminders(reminders)
    return {"reminders": reminders}
@app.get("/history")
def get_history():
    return {"history": conversation_history}


@app.get("/reset")
def reset():
    global conversation_history
    conversation_history = [
        {"role": "system", "content": "You are Benganson AI, a helpful personal assistant. You are not ChatGPT and you should never refer to yourself as ChatGPT."}
    ]
    save_history(conversation_history)
    return {"status": "Memory cleared"}

@app.get("/web-search")
def web_search(query: str):
    try:
        results = DDGS().text(query, max_results=5)
        summary = "\n\n".join([f"{r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        summary = "No results found."
        print("Search error:", e)

    conversation_history.append({
        "role": "user",
        "content": f"Search the web for: {query}\n\nHere are the top results:\n\n{summary}\n\nUsing these results, give me an accurate, up-to-date answer, and mention the source(s)."
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=conversation_history
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, I'm having trouble connecting right now. Please try again in a moment."
        print("Groq error:", e)

    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)

    return {
        "query": query,
        "reply": reply
    }
@app.get("/chat")
def chat(message: str):
    conversation_history.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=conversation_history
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "Sorry, I'm having trouble connecting right now. Please try again in a moment."
        print("Groq error:", e)

    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)

    return {
        "user_message": message,
        "reply": reply
    }