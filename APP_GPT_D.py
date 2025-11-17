import os
import re
import PyPDF2
import docx
import pandas as pd
import asyncio
from typing import List, Tuple, Any, Dict, Optional
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# Note : On utilise l'import standard qui fonctionne avec vos versions figées
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langdetect import detect, LangDetectException
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
FOLDER = "Docu_Cost_notTechnip" 
VECTOR_INDEX_PATH = "vector_index_cost_GPT"
RETRIEVER_K = 6
MAX_SOURCES_RETURN = 8

OPENAI_CHAT_MODEL = "gpt-5-nano-2025-08-07"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# Marqueurs
MARKER_FR = "Le contexte fourni n'a pas de rapport avec cette question."
MARKER_EN = "The provided context is not relevant to this question."

# ----------------------------
# DÉFINITION DES PROMPTS (VERSION SÉCURISÉE)
# On n'utilise PAS de f-string (f"") pour éviter de casser les variables {context}
# ----------------------------

# 1. FRANÇAIS
raw_template_fr = """Tu es un assistant IA. Réponds à la "Question" en te basant sur l'historique de la conversation et le "Contexte" fourni.
Règle cruciale : La réponse doit impérativement être en français.

MARQUEUR SPÉCIAL (À UTILISER TEL QUEL) :
Si le "Contexte" n'est pas pertinent, commence ta réponse par la phrase exacte : "MARKER_PLACEHOLDER" (avec l'espace à la fin).

INSTRUCTIONS :
1.  Priorité 1 : Utilise le "Contexte" ci-dessous pour répondre de manière factuelle à la "Question".
2.  Priorité 2 : Si le "Contexte" ne contient pas la réponse ou n'est pas pertinent, utilise le "MARQUEUR SPÉCIAL" puis réponds à la "Question" en utilisant tes connaissances générales et l'historique de la conversation.
3.  Priorité 3 : Si tu ne peux pas répondre du tout, dis-le poliment. N'invente pas une réponse.

Contexte:
{context}

Question:
{question}

Réponse:"""

# On injecte le marqueur via replace()
final_template_fr = raw_template_fr.replace("MARKER_PLACEHOLDER", MARKER_FR)

PROMPT_FR = PromptTemplate(
    template=final_template_fr, 
    input_variables=["context", "question"]
)

# 2. ANGLAIS
raw_template_en = """You are an AI assistant. Answer the "Question" based on the conversation history and the provided "Context".
Crucial rule: The answer must absolutely be in English.

SPECIAL MARKER (USE AS IS):
If the "Context" is not relevant, start your answer with the exact phrase: "MARKER_PLACEHOLDER" (with the space at the end).

INSTRUCTIONS:
1.  Priority 1: Use the "Context" below to factually answer the "Question".
2.  Priority 2: If the "Context" does not contain the answer or is not relevant, use the "SPECIAL MARKER" and then answer the "Question" using your general knowledge and the conversation history.
3.  Priority 3: If you cannot answer at all, state so politely. Do not invent an answer.

Context:
{context}

Question:
{question}

Answer:"""

final_template_en = raw_template_en.replace("MARKER_PLACEHOLDER", MARKER_EN)

PROMPT_EN = PromptTemplate(
    template=final_template_en, 
    input_variables=["context", "question"]
)

# 3. CONDENSATION (Simple string, pas de variables complexes ici)
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Étant donné l'historique de la conversation (Chat History) et une question de suivi (Follow Up Input), reformule la question de suivi pour qu'elle soit une question autonome, dans la même langue que la question de suivi.

Historique de la conversation (Chat History):
{chat_history}

Question de suivi (Follow Up Input): {question}

Directives:
- Si la question de suivi est déjà autonome, garde-la telle quelle.
- Si la question de suivi fait référence à une réponse précédente (par exemple "multiplie le résultat", "et pour lui ?", "fais-le * 2"), trouve la valeur ou le sujet dans l'historique (par exemple, la "réponse" de l'Assistant) et incorpore-la dans la nouvelle question.

Exemple:
Historique: [("combien font 10+10?", "La réponse est 20.")]
Question de suivi: "multiplie-le par 3"
Question autonome: "Que fait 20 * 3?"

Exemple 2:
Historique: [("Parle-moi de Paris", "Paris est la capitale de la France.")]
Question de suivi: "et pour Londres ?"
Question autonome: "Parle-moi de Londres"

Question autonome (Standalone question):"""
CONDENSE_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)


# ----------------------------
# Initialisation de FastAPI
# ----------------------------
app = FastAPI()

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Fix pour le logo (Favicon)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/images/logo.png")

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Any] = []
    session_id: Optional[str] = "global"

# ----------------------------
# Fonctions Utilitaires
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

# ----------------------------
# Cœur du Chatbot (Logique RAG)
# ----------------------------
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
    
    # --- CORRECTION TEMPERATURE ---
    # Le modèle gpt-5-nano impose temperature=1
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=1)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=CONDENSE_PROMPT
    )

# ----------------------------
# État de l'Application
# ----------------------------
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
        print("🤖 Instance de chatbot FRANÇAIS créée.")
        chatbot_instances['en'] = create_chatbot(vectorstore, PROMPT_EN)
        print("🤖 Instance de chatbot ANGLAIS créée.")

@app.on_event("startup")
async def startup_event():
    build_all_chatbots()
    global excels_global
    if os.path.exists(FOLDER):
        excel_files = [f for f in os.listdir(FOLDER) if f.endswith((".xlsx", ".xls")) and not f.startswith("~$")]
        for file in excel_files:
            excels_global[file] = load_excel(os.path.join(FOLDER, file))

# ----------------------------
# Routes
# ----------------------------
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
            print(f"ERREUR: {e}")
            return JSONResponse({"answer": "⚠️ Erreur interne.", "sources": []})

        answer = result.get("answer", "")
        sources = extract_unique_sources(result, max_sources=MAX_SOURCES_RETURN)
        
        answer_trimmed = answer.strip()
        if answer_trimmed.startswith(MARKER_FR) or answer_trimmed.startswith(MARKER_EN):
            sources = []
            
        return JSONResponse({"answer": answer, "sources": sources})

# ----------------------------
# MAIN (Pour Railway)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("APP_GPT_D:app", host="0.0.0.0", port=port)