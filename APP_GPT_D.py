import os
import shutil
import asyncio
from typing import List, Any, Dict, Optional
import uvicorn

# --- IMPORTS FASTAPI ---
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- IMPORTS LANGCHAIN ---
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langdetect import detect, LangDetectException
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
FOLDER = "Docu_Cost_notTechnip" 
VECTOR_INDEX_PATH = "vector_index_cost_GPT5N"
RETRIEVER_K = 6
MAX_SOURCES_RETURN = 8
OPENAI_CHAT_MODEL = "gpt-5-nano-2025-08-07"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

MARKER_FR = "Le contexte fourni n'a pas de rapport avec cette question."
MARKER_EN = "The provided context is not relevant to this question."

# --- PROMPTS ---
raw_template_fr = """Tu es un assistant IA. Réponds à la "Question" en te basant sur l'historique de la conversation et le "Contexte" fourni.
Règle cruciale : La réponse doit impérativement être en français.
MARQUEUR SPÉCIAL : Si le "Contexte" n'est pas pertinent, commence ta réponse par : "MARKER_PLACEHOLDER".
INSTRUCTIONS :
1. Priorité 1 : Utilise le "Contexte".
2. Priorité 2 : Si non pertinent, utilise le MARQUEUR puis tes connaissances.
Contexte:
{context}
Question:
{question}
Réponse:"""
final_template_fr = raw_template_fr.replace("MARKER_PLACEHOLDER", MARKER_FR)
PROMPT_FR = PromptTemplate(template=final_template_fr, input_variables=["context", "question"])

raw_template_en = """You are an AI assistant. Answer based on context.
Crucial rule: Answer in English.
SPECIAL MARKER: If context is irrelevant, start with: "MARKER_PLACEHOLDER".
Context:
{context}
Question:
{question}
Answer:"""
final_template_en = raw_template_en.replace("MARKER_PLACEHOLDER", MARKER_EN)
PROMPT_EN = PromptTemplate(template=final_template_en, input_variables=["context", "question"])

CONDENSE_PROMPT = PromptTemplate.from_template("""Reformule la question de suivi.\nHistorique:\n{chat_history}\nQuestion:\n{question}\nQuestion autonome:""")

# --- APP ---
app = FastAPI()

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/images/logo.png")

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Any] = []
    session_id: Optional[str] = "global"

chatbot_instances = {}
global_vectorstore = None 
locks_by_session = {}

def get_or_create_vector_db(index_path: str = VECTOR_INDEX_PATH):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    # 1. Essai chargement disque
    if os.path.exists(index_path):
        try:
            print(f"Chargement index '{index_path}'...")
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Erreur chargement: {e}")
    
    # 2. Fallback: Index vide pour permettre l'upload
    print("⚠️ Index introuvable. Création index vide.")
    dummy = Document(page_content="Init", metadata={"source": "system"})
    return FAISS.from_documents([dummy], embeddings)

def create_chatbot(vectorstore: FAISS, prompt: PromptTemplate):
    if not vectorstore: return None
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=CONDENSE_PROMPT
    )

def build_all_chatbots():
    global chatbot_instances, global_vectorstore
    if not os.getenv("OPENAI_API_KEY"): return
    global_vectorstore = get_or_create_vector_db()
    if global_vectorstore:
        chatbot_instances['fr'] = create_chatbot(global_vectorstore, PROMPT_FR)
        chatbot_instances['en'] = create_chatbot(global_vectorstore, PROMPT_EN)
        print("🤖 Chatbots prêts.")

@app.on_event("startup")
async def startup_event():
    build_all_chatbots()

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    global global_vectorstore
    if not global_vectorstore: return JSONResponse({"status": "error", "message": "Erreur init."}, status_code=500)

    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
        text = ""
        with open(temp_filename, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages: text += page.extract_text() or ""
        
        if not text.strip(): return JSONResponse({"status": "error", "message": "PDF vide/illisible."}, status_code=400)

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.create_documents([text], metadatas=[{"source": file.filename, "type": "user_upload"}])
        global_vectorstore.add_documents(chunks)
        
        return JSONResponse({"status": "success", "message": f"Document '{file.filename}' intégré !"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return HTMLResponse(f.read())
    return HTMLResponse("<h3>API Ready</h3>")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global chatbot_instances, locks_by_session
    query = (request.query or "").strip()
    if not query: return JSONResponse({"answer": "...", "sources": []})
    
    session_id = request.session_id or "global"
    lock = locks_by_session.setdefault(session_id, asyncio.Lock())

    async with lock:
        try: lang = detect(query)
        except: lang = 'en'
        chatbot = chatbot_instances.get(lang, chatbot_instances.get('en'))
        if not chatbot: return JSONResponse({"answer": "⚠️ Chatbot indisponible.", "sources": []})
        
        chat_tuples = []
        for item in request.chat_history[-6:]:
            if item.get("role") == "user": u = item.get("content")
            if item.get("role") == "assistant" and u: chat_tuples.append((u, item.get("content")))

        try:
            res = await asyncio.to_thread(chatbot.invoke, {"question": query, "chat_history": chat_tuples})
        except Exception as e:
            return JSONResponse({"answer": "Erreur interne.", "sources": []})

        answer = res.get("answer", "")
        sources = []
        seen = set()
        for d in res.get("source_documents", []):
            s_name = d.metadata.get("source", "Inconnu")
            s_page = d.metadata.get("page", "N/A")
            if (s_name, s_page) not in seen:
                seen.add((s_name, s_page))
                sources.append({"Document": s_name, "Page/Feuille": str(s_page)})

        if answer.strip().startswith(MARKER_FR) or answer.strip().startswith(MARKER_EN): sources = []
            
        return JSONResponse({"answer": answer, "sources": sources})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("APP_GPT_D:app", host="0.0.0.0", port=port)