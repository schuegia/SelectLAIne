# main.py
import sys, types
# Stub für fehlendes micropip
# .\venv\Scripts\activate
# Imports: pip install python-dotenv pdfplumber wandb langchain langchain-community langchain-core langchain-google-genai tqdm openai faiss-cpu tiktoken pandas openpyxl tika bs4 lxml pdfplumber pytesseract pillow
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
from tika import parser
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import io
import subprocess
import shlex


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
from langchain.schema import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
# 1) OCR Helper: nur auf Bilder anwenden
# —————————————————————————————————————————————————
def ocr_with_timeout(pil_img, lang="deu", timeout_s=2):
    """
    Speichert das PIL-Image temporär, ruft Tesseract per subprocess
    mit vollem Pfad auf und bricht nach timeout_s Sekunden ab.
    """
    # 1) Temp-Datei anlegen
    tmp_path = Path("ocr_tmp.png")
    pil_img.save(tmp_path)

    # 2) Pfad zur tesseract.exe ermitteln
    #    pytesseract.pytesseract.tesseract_cmd wurde ja schon gesetzt
    tesseract_exe = pytesseract.pytesseract.tesseract_cmd

    # 3) Kommando als Liste
    cmd = [
        tesseract_exe,
        str(tmp_path),
        "stdout",
        "-l", lang
    ]

    try:
        # 4) Aufruf mit Timeout
        res = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=timeout_s
        )
        return res.stdout.decode("utf-8")
    except subprocess.TimeoutExpired:
        return ""
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

def extract_image_only_ocr(path: str) -> str:
    """
    Extrahiert nur eingebettete Bilder aus dem PDF,
    cropt sie auf Seiten­grenzen und führt darauf
    Tesseract-OCR mit Timeout durch. Zeigt Fortschritt an.
    """
    ocr_texts = []
    filename = Path(path).name
    with pdfplumber.open(path) as pdf:
        total_imgs = sum(len(page.images) for page in pdf.pages)
        pbar = tqdm(total=total_imgs, desc=f"OCR {filename}")
        for page in pdf.pages:
            page_w, page_h = page.width, page.height
            for img in page.images:
                # Bounding Box clampen
                x0, y0 = max(0, img["x0"]), max(0, img["y0"])
                x1, y1 = min(page_w, img["x1"]), min(page_h, img["y1"])
                if x1 <= x0 or y1 <= y0:
                    pbar.update(1)
                    continue
                # Bild croppen und PIL-Image erzeugen
                cropped = page.crop((x0, y0, x1, y1))
                page_image = cropped.to_image(resolution=75)
                pil_img = page_image.original.convert("RGB")
                # OCR mit Timeout
                txt = ocr_with_timeout(pil_img, lang="deu", timeout_s=2)
                if txt.strip():
                    ocr_texts.append(f"[OCR-Bild]\n{txt.strip()}\n")
                pbar.update(1)
        pbar.close()
    return "\n".join(ocr_texts)

# —————————————————————————————————————————————————
# 2) Preprocessing der PDF-Dokumente - Tika-Parsing - Chunk entspricht immer genau einer Überschrift
# —————————————————————————————————————————————————

def pdf_to_xhtml(path: str) -> str:
    parsed = parser.from_file(path)
    return parsed.get('content', '')

def parse_xhtml_sections(xhtml: str) -> list[Document]:
    soup = BeautifulSoup(xhtml, 'lxml')
    docs = []
    current_title = None
    buffer = []
    for elem in soup.find_all(['h1','h2','h3','p']):
        if elem.name in ('h1','h2','h3'):
            if buffer:
                docs.append(Document(
                    page_content=(current_title or '') + "\n\n" + " ".join(buffer),
                    metadata={"source": current_title}
                ))
                buffer = []
            current_title = elem.get_text(strip=True)
        else:  # <p>
            text = elem.get_text(strip=True)
            if text:
                buffer.append(text)
    # letzten Puffer flushen
    if buffer:
        docs.append(Document(
            page_content=(current_title or '') + "\n\n" + " ".join(buffer),
            metadata={"source": current_title}
        ))
    return docs

# —————————————————————————————————————————————————
# 3) Setup & Index-Building (PDF→Chunks→FAISS)
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

# Strukturierte Chunks via Apache Tika
docs = []
for pdf in to_scan:
    # 0) OCR nur auf Bilder
    ocr_blob = extract_image_only_ocr(str(pdf))
    # Falls OCR-Text gefunden wurde, splitte ihn & indexiere
    if ocr_blob:
       for idx, chunk in enumerate(splitter.split_text(ocr_blob)):
        docs.append(Document(
        page_content=chunk,
        metadata={"source": f"{pdf.name}::OCR", "chunk": idx}   
        ))


    # 1) XHTML per Tika holen
    xhtml   = pdf_to_xhtml(str(pdf))
    # 2) Sections extrahieren
    sections = parse_xhtml_sections(xhtml)

    # 3) Jede Section in kleine Chunks splitten
    for sec in sections:
        # benutzt euren Splitter auf den reinen Text
        for idx, chunk in enumerate(splitter.split_text(sec.page_content)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":  f"{pdf.name}::{sec.metadata.get('source','')}",
                    "chunk":   idx
                }
            ))
    processed[pdf.name] = pdf.stat().st_mtime

# Manifest & Index-Ordner speichern
INDEX.mkdir(exist_ok=True)
MANIFEST.write_text(json.dumps(processed))

# FAISS laden oder neu erstellen (mit Guard gegen leere docs)
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Wenn ein schon gebauter Index existiert, immer zuerst laden
if (INDEX / "index.faiss").exists():
    db = FAISS.load_local(str(INDEX), emb, allow_dangerous_deserialization=True)
    # und nur bei neuen docs nachladen
    if docs:
        db.add_documents(docs)
        db.save_local(str(INDEX))
else:
    # Es existiert noch kein Index – baue nur, wenn Du docs hast
    if not docs:
        raise RuntimeError(
            "Kein FAISS-Index gefunden und auch keine Dokumente zum Indexieren. "
            "Bitte lege PDFs in 'data/' ab oder setze FULL_SCAN=true."
        )
    db = FAISS.from_documents(docs, emb)
    db.save_local(str(INDEX))

# —————————————————————————————————————————————————
# 4) Model, Retriever, Memory
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
# 5) Query-Transformation Prompt (für Branch & Pipeline)
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
# 6) Chat-Retriever mit RunnableBranch
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
# 7) Query Expansion & Reranking Setup
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
# 8) End-to-End Pipeline per Query mit HyDE
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
# 9) Testfrageset laden und mit einer Tabelle antworten - ansonsten Drei Beispiel-Queries
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
