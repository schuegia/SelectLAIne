# SelectLine ERP RAG System

## Project Description

Dieses Projekt implementiert ein Retrieval-Augmented Generation (RAG) System für die Dokumentation des SelectLine ERP-Systems. Es erlaubt Kunden und unserem Support-Personal per Chat Fragen über die Software zu stellen und eine KI-basierte Antwort, welche auf Handbücher und Schulungsunterlagen zugreift, mit Quellenangabe zu erhalten.

## Name & URL

| Name       | URL                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Code       | https://github.com/schuegia/SelectLAIne                                                                                             |
| Demo (optional) | (z. B. Streamlit oder Huggingface Space) — _wird aktuell nicht öffentlich gehostet_                                                   |
| Embeddings (Cloud)    | OpenAI Embeddings (`text-embedding-ada-002`) via `langchain_community.embeddings.OpenAIEmbeddings`                         |
| weitere getestete Embeddings (Lokal)    | Lokale SBERT-Modelle (z. B. [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html)) via `sentence-transformers` |

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

### Indexing
- **Structural Chunking**  
  Apache Tika→XHTML-Parsing von `<h1–h3>` & `<p>` → Überschriften werden erkannt und Chunks folgen echten Dokumentstrukturen - Sections nach Überschriften gegliedert statt starre Text-Blöcke  
- **Chunk Refinement**  
  Feinsplit in 1000 Zeichen mit 100 Zeichen Overlap  
- **Targeted OCR**  
  `pdfplumber` + `pytesseract` nur auf Screenshots → UI-Labels und Fehlermeldungen zuverlässig extrahiert  
- **Product Mapping**  
  Keyword-Map auf PDF-Quelldokumente, um Modul-Fokus zu stärken  

---

### Pre-Retrieval
- **Query Transformation (HyDE)**  
  Hypothetical Document Embeddings via fiktive Antwort → MMR-Search nach Pseudo-Dokumenten  
- **Query Expansion**  
  LLM-Chain generiert 3–5 Synonyme/Term-Varianten für bessere Recall-Rate  

---

### Retrieval
- **MMR Retrieval mit Cosine**  
  FAISS `max_marginal_relevance_search` mit Cosine-Distance (`k=5`, `fetch_k=40`, `λ=0.7`)  

---

### Post-Retrieval
- **Cross-Encoder Reranking**  
  LLMChain re-ordnet die Top-Hits nach echter Relevanz  
- **LLM-gestützte Selection**  
  Automatische Auswahl der 4 wichtigsten Chunks aus den Top-5  
- **Compression**  
  Zusammenfassung der finalen Chunks per LLM für kompakten Kontext  

---

### Generation
- **Generator-Verifikation**  
  LLM prüft, ob die Antwort vollständig durch den Kontext gedeckt ist und löst ggf. Fallback-Retrieval aus  


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

1. **Manuell Fragen** in `questions.xlsx` erstellt mit verschiedenen Schwerigkeitsgraden (Spalte `question`).  
2. **Import** via `pandas.read_excel` bzw. Fallback JSON.  
3. **Pipeline**: `process_query(q)` liefert Antwort + Quell-Chunk.  
4. **Ergebnis-Tabelle** in Konsole + `rag_results.json`.  
5. **Manuelle und automatisierte Bewertung** der Antworten (Manuell und von ChatGPT).

| Score | Beschreibung                                                                                              |
|-------|-----------------------------------------------------------------------------------------------------------|
| 0     | falsch (enthält keine oder irreführende Information)                                                      |
| 1     | teilweise korrekt (teils richtige, teils fehlende oder ungenaue Angaben)                                   |
| 2     | akzeptabel (keine unpassenden Inhalte)                                                                    |
| 3     | weitgehend korrekt (Antwort stimmt mit dem Handbuch überein)                                               |
| 4     | perfekt (inhaltlich und Umfang korrekt, Antwort würde einem Kunden weiterhelfen) 
---

## Results

| Lauf / Konfiguration                                                | Manual Avg. Accuracy (0–4) | Rating ChatGPT | Bemerkungen                                     |
|---------------------------------------------------------------------|----------------------------|----------------|-------------------------------------------------|
| Basis (fixed-chunks + OpenAI Embeddings)                            | 1.66                       | 2.0              | –                                               |
| + Product Mapping (Keyword-Map)                                      | 1.85                       | 2.3              | Verbesserte Wahl der Quellen                    |
| + HyDE                                                               | 2.10                       | 2.6              | Höhere Recall-Rate                              |
| + Post-Retrieval Selection & Compression                             | 1.54                       | 2.1              | Filterung zu restriktiv                         |
| + Auswahl der Chunks beim Compression erhöht + Generator-Verifikation | 2.32                       | 2.9              | Präzisere und komprimierte Outputs              |
| + Struktur-Chunks via Tika                                           | 2.68                       | 3.1              | Besserer Kontext-Bezug                          |
| + Targeted OCR                                                       | 2.51                       | 2.8              | UI-Labels und Screenshots abgedeckt  


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

Gian Schürch / SelectLine Software AG 