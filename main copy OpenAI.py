# main.py
import sys, types
# Stub für fehlendes micropip
# .\venv\Scripts\activate
# Imports: pip install python-dotenv pdfplumber wandb langchain langchain-community langchain-core langchain-google-genai tqdm openai faiss-cpu tiktoken
try:
    import micropip
except ImportError:
    sys.modules['micropip'] = types.ModuleType('micropip')

import os, json, re
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber
from tqdm import tqdm
import wandb
import openai
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),  # fügt die Chat-History als Platzhalter ein
    ]
)

# Produkt-Zuordnung: Schlüsselwort → PDF-Source
product_mapping = {
    "SelectLine Auftrag Handbuch CH Aktuelle Version (1).pdf": [
        "auftrag", "aufträge", "auftragsverwaltung", "auftragserfassung", "belegerfassung", "beleg", "warenwirtschaft", "artikelstamm", "artikelverwaltung"
    ],
    "SelectLine Rechnungswesen Handbuch CH Aktuelle Version.pdf": [
        "rechnungswesen", "buchhaltung", "finanzbuchhaltung", "debitoren", "kreditoren", "kontenplan", "buchen"
    ],
    "SelectLine Produktion Handbuch CH Aktuelle Version.pdf": [
        "produktion", "fertigungsauftrag", "fertigung"
    ],
    "SelectLine Lohn Handbuch CH Aktuelle Version.pdf": [
        "lohn", "lohnbuchhaltung", "swissdec", "lohndaten", "Quellensteuer"
    ],
    "SelectLine Kassabuch Handbuch CH Aktuelle Version.pdf": [
        "kassenbuch", "kassabuch"
    ],
    "SelectLine Mobile Handbuch CH Aktuelle Version.pdf": [
        "mobile", "app", "mobile app", "smartphone"
    ],
    "SelectLine CRM Handbuch CH Aktuelle Version.pdf": [
        "crm", "CRM"
    ],
    "Schulungsunterlagen Auftrag Einsteiger.pdf": [
        "auftrag", "aufträge", "auftragsverwaltung", "auftragserfassung", "belegerfassung", "beleg", "warenwirtschaft", "artikelstamm", "artikelverwaltung"
    ],
    "Schulungsunterlagen Auftrag Fortgeschritten.pdf": [
        "auftrag", "aufträge", "auftragsverwaltung", "auftragserfassung", "belegerfassung", "beleg", "warenwirtschaft", "artikelstamm", "artikelverwaltung"
    ],
    "Schulungsunterlagen Auftrag Profi.pdf": [
        "auftrag", "aufträge", "auftragsverwaltung", "auftragserfassung", "belegerfassung", "beleg", "warenwirtschaft", "artikelstamm", "artikelverwaltung"
    ],
    "Schulungsunterlagen Fibu.pdf": [
        "fibu", "rechnungswesen", "buchhaltung", "finanzbuchhaltung", "debitoren", "kreditoren", "kontenplan", "buchen"
    ],
    "Schulungsunterlagen CRM.pdf": [
        "crm", "CRM"
    ],
    "Schulungsunterlagen Dashboard.pdf": [
        "dashboard", "widget"
    ],
    
}

# —————————————————————————————————————————————————
# 1) Setup & Index-Building (PDF→Chunks→FAISS)
# —————————————————————————————————————————————————
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE = Path(__file__).parent
DATA = BASE / "data"
INDEX = BASE / "faiss_index"
MANIFEST = INDEX / "processed_files.json"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def extract_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            parts.append(txt + "\n")
    joined = "".join(parts).strip()
    return re.sub(r"\s+", " ", joined)

# Laden/Initialisieren Manifest
if MANIFEST.exists():
    processed = json.loads(MANIFEST.read_text())
else:
    processed = {}
if os.getenv("FULL_SCAN","").lower() in ("1","true","yes"):
    processed = {}

# Neue/aktualisierte PDFs ermitteln
to_scan = [
    p for p in DATA.glob("*.pdf")
    if p.name not in processed or processed[p.name] < p.stat().st_mtime
]

# Chunks erzeugen
docs = []
for pdf in to_scan:
    txt = extract_text(str(pdf))
    for idx, chunk in enumerate(splitter.split_text(txt)):
        docs.append(Document(page_content=chunk,
                             metadata={"source": pdf.name, "chunk": idx}))
    processed[pdf.name] = pdf.stat().st_mtime

# Manifest & Index-Ordner speichern
INDEX.mkdir(exist_ok=True)
MANIFEST.write_text(json.dumps(processed))

# FAISS laden oder neu erstellen
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
if (INDEX / "index.faiss").exists():
    db = FAISS.load_local(str(INDEX), emb, allow_dangerous_deserialization=True)
    if docs:
        db.add_documents(docs)
        db.save_local(str(INDEX))
else:
    db = FAISS.from_documents(docs, emb)
    db.save_local(str(INDEX))

# —————————————————————————————————————————————————
# 2) Model, Retriever, Memory
# —————————————————————————————————————————————————
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 40,
        "lambda_mult": 0.7,
        "distance_metric": "cosine"
    }
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# —————————————————————————————————————————————————
# 3) Query-Transformation Prompt (für Branch & Pipeline)
# —————————————————————————————————————————————————
query_transform_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    (
        "user",
        "Given the above conversation, generate one precise search query. "
        "Only output the query."
    ),
])

# —————————————————————————————————————————————————
# 4) Chat-Retriever mit RunnableBranch
#    (erster Aufruf: direkte Nutzung,
#     Folgefragen: transform → retrieve)
# —————————————————————————————————————————————————
query_transforming_retriever_chain = RunnableBranch(
    (
        lambda inp: len(inp.get("messages", [])) == 1,
        # Bei nur 1 Message: nutze content direkt
        (lambda inp: inp["messages"][-1].content) | retriever,
    ),
    # Sonst: transform → parse → retrieve
    query_transform_prompt | model | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

# —————————————————————————————————————————————————
# 5) Query Expansion & Reranking Setup
# —————————————————————————————————————————————————
expansion_prompt = PromptTemplate(
    template="""
Given a search query, list 3–5 German synonyms or related phrases, comma-separated.
Query: {query}
""",
    input_variables=["query"],
)
expander = LLMChain(llm=model, prompt=expansion_prompt)

rerank_prompt = ChatPromptTemplate.from_template(
    """User query: {query}
Rank these chunks by relevance; output comma-separated indices only:
{docs}
"""
)
rerank_chain = LLMChain(llm=model, prompt=rerank_prompt)

# —————————————————————————————————————————————————
# 6) End-to-End Pipeline per Query mit HyDE
# —————————————————————————————————————————————————
def process_query(user_q: str):
    # a) Chat-History aktualisieren
    history = memory.load_memory_variables({})["chat_history"]
    history.append(HumanMessage(content=user_q))

    # b) Chat-Retrieval: Branch entscheidet, ob transform nötig
    candidates = query_transforming_retriever_chain.invoke({"messages": history})

    # c) Query Expansion
    base_q = history[-1].content
    expansions = expander.predict(query=base_q)
    terms = [t.strip() for t in expansions.split(",") if t.strip()]
    all_qs = terms + [base_q]

    # — HyDE: Hypothetische Dokument-Embedding ergänzen —
    # 1) Fiktive Antwort erzeugen
    hyde_prompt = PromptTemplate(
        template="""
Given the user question, write a brief, factual paragraph that answers it as if from the documentation:

Question:
{question}

Hypothetical Answer:
""",
        input_variables=["question"]
    )
    # 1) Baue einen HyDE-LLMChain
    hyde_chain = LLMChain(llm=model, prompt=hyde_prompt)

    # 2) Erzeuge die Hypothetical Answer
    pseudo_doc = hyde_chain.predict(question=base_q)
    # 2) Pseudo-Dokument embedd­en
    pseudo_emb = emb.embed_query(pseudo_doc)
    # 3) HyDE-Retrieval (MMR) zusätzlicher Kandidaten
    hyde_docs = db.max_marginal_relevance_search_by_vector(
        pseudo_emb,
        k=5,
        fetch_k=20,
        lambda_mult=0.7,
        distance_metric="cosine"
    )

    # d) Sammle alle Kandidaten für jede erweiterte Query + HyDE
    all_docs = []
    for qexp in all_qs:
        all_docs.extend(retriever.get_relevant_documents(qexp))
    all_docs.extend(hyde_docs)

    # Duplikate entfernen
    unique = list({d.page_content: d for d in all_docs}.values())

    # e) Reranken via Rerank-Chain
    docs_str = "\n".join(
        f"[{i}] {d.metadata.get('source','unknown')} "
        f"(chunk {d.metadata.get('chunk','?')}): {d.page_content[:80]}..."
        for i, d in enumerate(unique)
    )
    ranking = rerank_chain.predict(query=base_q, docs=docs_str)

    # 1) Alle per Regex gefundenen Indizes in Integers umwandeln
    all_idxs = [int(i) for i in re.findall(r"\d+", ranking)]
    # 2) Nur diejenigen behalten, die innerhalb des Bereichs liegen
    valid_idxs = [i for i in all_idxs if 0 <= i < len(unique)]
    # 3) Maximal k=5 verwenden
    valid_idxs = valid_idxs[:5]

    # 4) top5 auf Basis der validen Indizes oder Fallback zu den ersten 5
    if valid_idxs:
        top5 = [unique[i] for i in valid_idxs]
    else:
        top5 = unique[:5]


    # f) Post-Retrieval Compression & Selection
    top_texts = [d.page_content for d in top5]

    # A) LLM-gestützte Selection (4 von 5)
    select_prompt = PromptTemplate(
        template="""
Du bekommst 5 Text-Abschnitte. Wähle die 4 Abschnitte aus, die am relevantesten zur Frage sind.
Frage: {question}

Abschnitte:
{chunks}

Gib als Antwort nur die Indizes (0–4), kommagetrennt.
""",
        input_variables=["question", "chunks"]
    )
    select_chain = LLMChain(llm=model, prompt=select_prompt)
    idxs_str = select_chain.predict(
        question=base_q,
        chunks="\n\n".join(f"[{i}] {text[:80]}…" for i, text in enumerate(top_texts))
    )
    sel_idxs = [int(i) for i in re.findall(r"\d+", idxs_str)]
    sel_idxs = [i for i in sel_idxs if 0 <= i < len(top_texts)][:4]
    if sel_idxs:
        top_texts = [top_texts[i] for i in sel_idxs]

    # B) Compression der finalen Chunks
    compress_prompt = PromptTemplate(
        template="""
Fasse die folgenden Text-Abschnitte knapp und verständlich zusammen, damit ein Kunde sie als Kontext erhält:

{chunks}

Kurze Zusammenfassung:
""",
        input_variables=["chunks"]
    )
    compress_chain = LLMChain(llm=model, prompt=compress_prompt)
    summary = compress_chain.predict(chunks="\n\n".join(top_texts))

    # C) Zusammenfassung als Context benutzen
    context = summary

    # D) Antwort generieren
    answer_chain = LLMChain(
        llm=model,
        prompt=PromptTemplate(
            template="""
Answer the user’s question based solely on the context below.
If the answer isn’t contained in the context, say “Dazu habe ich keine Information.”

Context:
{context}

Question: {question}
Answer:
""",
            input_variables=["context", "question"],
        ),
    )
    ans = answer_chain.predict(context=context, question=user_q)

    # g) Generator-Verifikation
    verify_prompt = PromptTemplate(
        template="""
Du bekommst Frage, Kontext und Antwort. Prüfe, ob die Antwort korrekt und vollständig durch den Kontext gestützt wird.
Gib als Ausgabe ein kurzes JSON-Objekt mit:
  - "valid": true|false
  - "explanation": kurze Begründung

Frage: {question}

Kontext:
{context}

Antwort:
{answer}

JSON:
""",
        input_variables=["question", "context", "answer"],
    )
    verify_chain = LLMChain(llm=model, prompt=verify_prompt)
    verification = verify_chain.predict(
        question=base_q,
        context=context,
        answer=ans
    )
    import json as _json
    try:
        v = _json.loads(verification)
    except Exception:
        v = {"valid": False, "explanation": verification}

    # Fallback bei ungültiger Verifikation
    if not v.get("valid", False):
        more_docs = db.max_marginal_relevance_search_by_vector(
            emb.embed_query(base_q), k=5, fetch_k=40, lambda_mult=0.5, distance_metric="cosine"
        )
        more_context = context + "\n\n" + "\n\n".join(d.page_content for d in more_docs)
        ans = answer_chain.predict(context=more_context, question=user_q)

    return ans, top5

# —————————————————————————————————————————————————
# 8) Testfrageset laden und mit einer Tabelle antworten - ansonsten Drei Beispiel-Queries
# —————————————————————————————————————————————————
# Berechne BASE wie oben:
BASE = Path(__file__).parent

# 1) Excel-Pfad definieren
excel_path = BASE / "questions.xlsx"
json_path  = BASE / "questions.json"

# 2) Fragen laden: Excel hat eine Spalte "question"
if excel_path.exists():
    df = pd.read_excel(excel_path, engine="openpyxl")
    questions = df["question"].dropna().astype(str).tolist()
elif json_path.exists():
    with open(json_path, encoding="utf-8") as f:
        questions = json.load(f)
else:
    questions = []  # oder Fallback-Liste

results = []
for q in questions:
    ans, sources = process_query(q)
    if sources:
        src   = sources[0].metadata["source"]
        chunk = sources[0].metadata["chunk"]
    else:
        src, chunk = "n/a", -1
    results.append({
        "question": q,
        "answer":   ans,
        "source":   src,
        "chunk":    chunk,
    })

# 4) Konsolen-Ausgabe
print(f"{'Frage':<60} {'Antwort':<40} {'Quelle':<20} {'Chunk':<5}")
print("-"*130)
for r in results:
    print(f"{r['question'][:60]:<60} {r['answer'][:40]:<40} {r['source']:<20} {r['chunk']:<5}")

# 5) Optional: Ergebnisse als JSON speichern
out = BASE / "rag_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nErgebnisse gespeichert in: {out}")


wandb.init(project="erp-rag", name="example_run")
for q in examples:
    ans, sources = process_query(q)
    print(f"\nQ: {q}\nA: {ans}\nSources:")
    for d in sources:
        src   = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk", "?")
        print(f" - {src}#chunk{chunk}")
    wandb.log({"query": q, "answer": ans})
wandb.finish()