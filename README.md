# 🧠 AI Data Modeling Agent

An interactive Streamlit application that assists with database schema design using a Large Language Model (LLM) and a **Human-in-the-Loop (HITL)** workflow. The agent supports two input modes — natural language business descriptions and structured CSV files — and exports the resulting data model as an ERD diagram, SQL DDL, and a comprehensive PDF audit report.

Built as part of a Master's thesis at the Prague University of Economics and Business (VŠE), Faculty of Informatics and Statistics.

---

## ✨ Features

- **Business mode** — paste a natural language business description and let the agent propose entities, attributes, and relationships
- **CSV mode** — upload CSV files and let the agent infer a relational schema from your data
- **HITL validation** — review and edit every step: entities, attributes, relationships, and primary keys
- **ERD diagram** — live Mermaid ERD rendered in the browser
- **SQL DDL export** — download the final schema as a `.sql` file
- **PDF audit report** — full session report with timeline, ER model, relational schema, ERD image, and SQL DDL (with Czech diacritics support via Noto Sans fonts)
- **Audit logging** — every agent and user action is timestamped and stored in session state

---

## 🗂️ Project Structure

```
├── AI_Agent_app.py          # Main Streamlit application
├── NotoSans-Regular.ttf     # Font for PDF export (Czech diacritics)
├── NotoSans-Bold.ttf        # Font for PDF export (bold)
├── NotoSansMono-Regular.ttf # Font for PDF export (monospace, optional)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9+
- A [Hugging Face](https://huggingface.co/) account with an API token
- Internet access (for Mermaid diagram rendering via `mermaid.ink`)

---

## 🚀 Installation & Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/soleka00/AI_Agent
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your Hugging Face API token**

Create a `.env` file or set it as an environment variable:
```bash
export HF_TOKEN=your_huggingface_token_here
```

Or add it directly in the Streamlit sidebar when the app starts.

**4. Run the app**
```bash
streamlit run AI_Agent_app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
huggingface-hub
reportlab
requests
```

Install all at once:
```bash
pip install streamlit pandas numpy huggingface-hub reportlab requests
```

---

## 🖥️ Deployment on Hugging Face Spaces

This app is designed to run on [Hugging Face Spaces](https://huggingface.co/spaces) with the Streamlit SDK.

1. Create a new Space with **Streamlit** as the SDK
2. Upload all files including the `.ttf` font files
3. Add your `HF_TOKEN` as a **Space secret** in Settings → Variables and secrets
4. The app will build and launch automatically

> ⚠️ Font files (`NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`) must be present in the root of the repository for Czech diacritics to render correctly in PDF exports. Without them, the app falls back to Helvetica.

---

## 🔄 How It Works

### Business Mode (Text Input)
1. Enter a natural language business description
2. Agent proposes entities, attributes, and relationships (ER model)
3. **HITL Step 1** — review and approve/edit entities
4. **HITL Step 2** — review and approve/edit relationships
5. Agent generates relational schema (tables, columns, PKs, FKs)
6. **HITL Step 3** — validate and edit the final schema
7. Export ERD diagram, SQL DDL, Blueprint JSON, and PDF audit report

### CSV Mode (Structured Input)
1. Upload one or more CSV files
2. Agent profiles the data and suggests column types and primary keys
3. **Step 1** — LLM-assisted schema patch (optional, scoped to selected tables)
4. **Step 2** — Human validation of schema via editable table
5. **Step 3** — Review and approve relationship candidates
6. **Step 4** — Generate and download ERD and SQL DDL

---

## 📄 Output Artifacts

| Artifact | Format | Description |
|---|---|---|
| ERD Diagram | Mermaid / PNG | Entity-Relationship Diagram |
| SQL DDL | `.sql` | CREATE TABLE statements with PKs and FKs |
| Blueprint JSON | `.json` | Full structured model (business mode) |
| Audit Report | `.pdf` | Complete session log with all steps and outputs |

---

## 📝 Notes

- The LLM used is accessed via [Hugging Face Inference API](https://huggingface.co/inference-api). The model can be configured inside the app sidebar.
- JSON repair is built-in — if the LLM returns malformed JSON, the app attempts automatic repair before failing.
- The Mermaid ERD is rendered client-side using [mermaid.js](https://mermaid.js.org/) via CDN and converted to PNG for PDF export via [mermaid.ink](https://mermaid.ink/).

---

## 👩‍💻 Author

Ekaterina Solyanik  
Master's thesis — Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE)  
2025–2026
