# SelectLine ERP RAG System

## Project Description

Dieses Projekt implementiert ein Retrieval-Augmented Generation (RAG) System für die Dokumentation des SelectLine ERP-Systems. Es erlaubt Kunden und unserem Support-Personal per Chat Fragen über die Software zu stellen und eine KI-basierte Antwort, welche auf Handbücher und Schulungsunterlagen zugreift, mit Quellenangabe zu erhalten.

## Name & URL

| Name       | URL                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Code       | https://github.com/schuegia/SelectLAIne                                                                                             |
| Demo (optional) | (z. B. Streamlit oder Huggingface Space) — _wird aktuell nicht öffentlich gehostet_                                                   |
| Embedding| "text-embedding-ada-002" von OpenAIEmbedding |

---

## Data Sources

| Data Source                                                                                 | Beschreibung                                           |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `data/Schulungsunterlagen Auftrag Einsteiger.pdf`                                           | Schulungsunterlagen Einsteiger – Modul Auftrag         |
| `data/Schulungsunterlagen Auftrag Fortgeschritten.pdf`                                      | Schulungsunterlagen Fortgeschritten – Modul Auftrag    |
| `data/Schulungsunterlagen Auftrag Profi_.pdf`                                              | Schulungsunterlagen Profi – Modul Auftrag              |
| `data/Schulungsunterlagen CRM.pdf`                                                          | Schulungsunterlagen – Modul CRM                        |
| `data/Schulungsunterlagen Dashboard.pdf`                                                    | Schulungsunterlagen – Dashboard & Reporting            |
| `data/Schulungsunterlagen Formulareditor.pdf`                                              | Schulungsunterlagen – Formular-Editor                  |
| `data/Schulungsunterlagen Lohn mit swissdec5.pdf`                                          | Schulungsunterlagen – Modul Lohnbuchhaltung (Swissdec)|
| `data/Schulungsunterlagen Makroassistent.pdf`                                              | Schulungsunterlagen – Makro-Assistent                  |
| `data/Schulungsunterlagen Toolbox.pdf`                                                     | Schulungsunterlagen – Toolbox & Add-ons                |
| `data/Schulungsunterlagen Fibu.pdf`                                                        | Schulungsunterlagen – Finanzbuchhaltung (Fibu)         |
| `data/SelectLine Auftrag Handbuch CH Aktuelle Version (1).pdf`                             | Offizielles Handbuch – Modul Auftrag                   |
| `data/SelectLine CRM Handbuch CH Aktuelle Version.pdf`                                     | Offizielles Handbuch – Modul CRM                       |
| `data/SelectLine Kassabuch Handbuch CH Aktuelle Version.pdf`                               | Offizielles Handbuch – Modul Kassabuch                 |
| `data/SelectLine Lohn Handbuch CH Aktuelle Version.pdf`                                     | Offizielles Handbuch – Modul Lohnbuchhaltung           |
| `data/SelectLine Mobile Handbuch CH Aktuelle Version.pdf`                                   | Offizielles Handbuch – Modul Mobile App                |
| `data/SelectLine Produktion Handbuch CH Aktuelle Version.pdf`                               | Offizielles Handbuch – Modul Produktion/Fertigung      |
| `data/SelectLine Rechnungswesen Handbuch CH Aktuelle Version.pdf`                           | Offizielles Handbuch – Modul Rechnungswesen            |
| `data/SelectLine System Handbuch CH Aktuelle Version.pdf`                                   | Offizielles Handbuch – System- und Konfigurations-Module|


---

## RAG Improvements

| Improvement                               | Beschreibung                                                                                                 |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Structural Chunking                       | Apache Tika → XHTML-Parsing von `<h1–h3>` & `<p>` → echte Sections statt starrer Zeichen-Blöcke              |
| Chunk Refinement                          | Post-Tika Split in 1000 Zeichen mit 100 Zeichen Overlap                                                      |
| Targeted OCR                              | `pdfplumber` + `pytesseract` nur auf Screenshots → UI-Labels und Fehlermeldungen werden zuverlässig erkannt |
| Product Mapping                           | Keyword-Mapping auf PDF-Quelle, um Modul-Fokus zu verstärken                                                   |
| Query Transformation (HyDE)               | Hypothetical Document Embeddings via fiktive Antwort → MMR Search nach Pseudo-Docs                           |
| Query Expansion                           | LLM-Chain generiert 3–5 Synonyme/Term-Varianten für bessere Recall                                           |
| Retrieval via MMR + Cosine                | FAISS `max_marginal_relevance_search` mit Cosine-Distance                                                    |
| Cross-Encoder Reranking                   | LLMChain ordnet Top-Hits neu                                                                                  |
| LLM-gestützte Selection                   | Auswahl der 4 relevantesten Chunks aus den Top-5                                                              |
| Compression                               | LLM-Chain fasst die finalen Chunks zusammen                                                                  |
| Generator-Verifikation                    | LLM-Prompt überprüft Antwort mit Kontext und löst ggf. Fallback Retrieval aus                                |

---

## Chunking

| Typ                        | Konfiguration                             |
|----------------------------|-------------------------------------------|
| Section-Based (Apache Tika)| Überschrift + Absatz als ein Chunk        |
| Zeichen-Splitter           | 1000 Chars, 100 Overlap per Section       |

---

## Choice of LLM

| Name                | Link                                                                                     |
|---------------------|------------------------------------------------------------------------------------------|
| Google Gemini 1.5 Pro | https://ai.google.dev/gemini-api/docs/models#gemini-1.5-pro-latest                      |

---

## Test Method

1. **Fragen** in `questions.xlsx` (Spalte `question`) gepflegt.  
2. **Import** via `pandas.read_excel` bzw. Fallback JSON.  
3. **Pipeline**: `process_query(q)` liefert Antwort + Quell-Chunk.  
4. **Ergebnis-Tabelle** in Konsole + `rag_results.json`.  
5. **Manuelle und automatisierte Bewertung** der Antworten (Numerische Scores, Generator-Verifikation).

---

## Results

| Lauf / Konfiguration                        | Avg. Accuracy (0–4) | Bemerkungen                       |
|---------------------------------------------|---------------------|-----------------------------------|
| Basis (fixed-chunks + OpenAI Embeddings)    | –                   | –                                 |
| + Struktur-Chunks via Tika                  | –                   | Besserer Kontext-Bezug            |
| + HyDE                                      | –                   | Höhere Recall-Rate                |
| + Post-Retrieval Selection & Compression    | –                   | Präzisere und komprimierte Outputs|
| + Targeted OCR                              | –                   | UI-Labels und Screenshots abgedeckt|

_Fülle die Werte nach Deinem Evaluations-Workflow ein._

---

## References

- **LangChain** (0.x) – Chaining, Retrieval, Memory  
- **Apache Tika** – PDF → XHTML Structure-Parsing  
- **pdfplumber**, **pytesseract** – Targeted OCR  
- **FAISS-CPU** – Vektor-Index  
- **Sentence-Transformers** – lokale Embeddings (optional)  
- **wandb** – Experiment Tracking  
- **dotenv**, **openai**, **langchain_google_genai**  

---

## License

MIT © Dein Name / SelectLine Software AG 