import os
import re
import PyPDF2
import docx
import pandas as pd
import asyncio
from typing import List, Tuple, Any, Dict, Optional
import uvicorn

from fastapi import FastAPI, Request
# --- AJOUT : Import nécessaire pour renvoyer une image
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# Configuration
FOLDER = "Docu_Cost_notTechnip" 
VECTOR_INDEX_PATH = "vector_index_cost_GPT"
RETRIEVER_K = 6
MAX_SOURCES_RETURN = 8
OPENAI_CHAT_MODEL = "gpt-5-nano-2025-08-07"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# Marqueurs
MARKER_FR = "Le contexte fourni n'a pas de rapport avec cette question."
MARKER_EN = "The provided context is not relevant to this question."

# Prompts (Je les garde condensés pour la lisibilité, ils sont identiques à avant)
PROMPT_FR_TEMPLATE = f"""Tu es un assistant IA... (votre prompt habituel)... Réponse:"""
PROMPT_FR = PromptTemplate(template=PROMPT_FR_TEMPLATE, input_variables=["context", "question"])

PROMPT_EN_TEMPLATE = f"""You are an AI assistant... (votre prompt habituel)... Answer:"""
PROMPT_EN = PromptTemplate(template=PROMPT_EN_TEMPLATE, input_variables=["context", "question"])

CONDENSE_QUESTION_PROMPT_TEMPLATE = """Étant donné l'historique... (votre prompt habituel)..."""
CONDENSE_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

# ----------------------------
# Initialisation de FastAPI & FAVICON FIX
# ----------------------------
app = FastAPI()

# 1. On monte le dossier static pour le CSS et JS
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. --- LA SOLUTION MAGIQUE POUR LE LOGO ---
# Si le navigateur demande /favicon.ico, on lui envoie directement le fichier logo.png
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Assurez-vous que le chemin est correct sur votre PC
    return FileResponse("static/images/logo.png")

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Any] = []
    session_id: Optional[str] = "global"

# ... (Le reste de vos fonctions utilitaires : extract_docs, load_documents, etc. restent inchangées) ...
# Je ne les remets pas ici pour ne pas faire un message de 3km, mais 
# GARDEZ TOUT LE RESTE DE VOTRE CODE (fonctions et endpoints) IDENTIQUE.

# ... (Vos fonctions extract_docs_from_pdf, etc.) ...

# ----------------------------
# Fonctions Utilitaires (Rappel pour copier-coller, assurez-vous de les avoir)
# ----------------------------
def extract_docs_from_pdf(path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if getattr(reader, "is_encrypted", False):
                try: reader.decrypt("")
                except Exception: return docs
            for i, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                if txt.strip():
                    docs.append(Document(page_content=txt, metadata={"source": os.path.basename(path), "page": i}))
    except Exception as e:
        print(f"⚠️ Erreur lecture PDF {path}: {e}")
    return docs

def extract_docs_from_docx(path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        d = docx.Document(path)
        paragraphs = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        if paragraphs:
            docs.append(Document(page_content="\n".join(paragraphs), metadata={"source": os.path.basename(path), "page": 1}))
    except Exception as e:
        print(f"⚠️ Erreur lecture DOCX {path}: {e}")
    return docs

def load_documents(folder: str) -> List[Document]:
    all_docs: List[Document] = []
    if not os.path.exists(folder): return all_docs
    files = [f for f in os.listdir(folder) if f.lower().endswith((".pdf", ".docx"))]
    for file in files:
        full_path = os.path.join(folder, file)
        docs = extract_docs_from_pdf(full_path) if file.lower().endswith(".pdf") else extract_docs_from_docx(full_path)
        all_docs.extend(docs)
    print(f"chargé {len(all_docs)} pages/documents.")
    return all_docs

def load_excel(path: str) -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(path)
        return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names}
    except Exception as e:
        print(f"⚠️ Erreur lecture Excel {path} : {e}")
        return {}

def analyse_excel(query: str, excels: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Optional[str], Optional[List[Dict[str, str]]]]:
    return None, None

def build_chat_history_for_chain(chat_history_raw: List[Any], max_turns: int = 6) -> List[Tuple[str, str]]:
    if not chat_history_raw: return []
    pairs: List[Tuple[str, str]] = []
    last_user = None
    for item in chat_history_raw:
        if isinstance(item, dict):
            role = item.get("role", "").lower()
            content = item.get("content", "")
            if role == "user": last_user = content
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, content))
                last_user = None
    return pairs[-max_turns:]

def extract_unique_sources(result_obj: Dict[str, Any], max_sources: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    docs = result_obj.get("source_documents", [])
    if not docs: return out
    for d in docs:
        meta = d.metadata
        src_norm = meta.get("source", "inconnu")
        page = meta.get("page", "?")
        key = (src_norm, page)
        if key not in seen:
            seen.add(key)
            out.append({"Document": src_norm, "Page/Feuille": str(page)})
            if len(out) >= max_sources: break
    return out

def get_or_create_vector_db(docs: List[Document], index_path: str = VECTOR_INDEX_PATH):
    if not docs: return None
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    if os.path.exists(index_path):
        try:
            print(f"Chargement de l'index FAISS depuis '{index_path}'...")
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Problème de chargement: {e}")
    print("Création d'un nouvel index FAISS...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def create_chatbot(vectorstore: FAISS, prompt: PromptTemplate):
    if not vectorstore: return None
    # TEMPERATURE DOIT ETRE 1 POUR CE MODELE
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=CONDENSE_PROMPT
    )

chatbot_instances: Dict[str, ConversationalRetrievalChain] = {}
excels_global: Dict[str, Dict[str, pd.DataFrame]] = {}
locks_by_session: Dict[str, asyncio.Lock] = {}

def build_all_chatbots():
    global chatbot_instances
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERREUR CRITIQUE : La variable d'environnement OPENAI_API_KEY n'est pas définie.\n")
        return
    docs = load_documents(FOLDER)
    vectorstore = get_or_create_vector_db(docs)
    if vectorstore:
        chatbot_instances['fr'] = create_chatbot(vectorstore, PROMPT_FR)
        chatbot_instances['en'] = create_chatbot(vectorstore, PROMPT_EN)

@app.on_event("startup")
async def startup_event():
    build_all_chatbots()
    global excels_global
    if os.path.exists(FOLDER):
        excel_files = [f for f in os.listdir(FOLDER) if f.endswith((".xlsx", ".xls")) and not f.startswith("~$")]
        for file in excel_files:
            excels_global[file] = load_excel(os.path.join(FOLDER, file))

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>API RAG & Excel ready</h3>")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global chatbot_instances, excels_global, locks_by_session
    query = (request.query or "").strip()
    if not query: return JSONResponse({"answer": "Veuillez poser une question.", "sources": []})
    session_id = request.session_id or "global"
    lock = locks_by_session.setdefault(session_id, asyncio.Lock())

    async with lock:
        excel_answer, excel_sources = analyse_excel(query, excels_global)
        if excel_answer: return JSONResponse({"answer": excel_answer, "sources": excel_sources or []})
        
        try: lang = detect(query)
        except LangDetectException: lang = 'en'
        
        chatbot_instance = chatbot_instances.get(lang, chatbot_instances.get('en'))
        if not chatbot_instance: return JSONResponse({"answer": "⚠️ Chatbot RAG non initialisé.", "sources": []})
        
        chat_history_tuples = build_chat_history_for_chain(request.chat_history, max_turns=6)
        try:
            result = await asyncio.to_thread(chatbot_instance.invoke, {"question": query, "chat_history": chat_history_tuples})
        except Exception as e:
            return JSONResponse({"answer": "⚠️ Erreur interne.", "sources": []})

        answer = result.get("answer", "")
        sources = extract_unique_sources(result, max_sources=MAX_SOURCES_RETURN)
        answer_trimmed = answer.strip()
        if answer_trimmed.startswith(MARKER_FR) or answer_trimmed.startswith(MARKER_EN): sources = []
        return JSONResponse({"answer": answer, "sources": sources})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("APP_GPT_D:app", host="0.0.0.0", port=port)