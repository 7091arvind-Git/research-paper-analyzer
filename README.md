# ResearchMate – Agentic RAG for Research Paper Analysis

ResearchMate is a simple, Python-based Agentic Retrieval-Augmented Generation (RAG) project designed to analyze scientific research papers. 

Instead of relying on a single LLM prompt, ResearchMate uses an agentic workflow:
1. **Retriever Agent**: Finds relevant sections from the paper using FAISS vector search.
2. **Research Analyst Agent**: Generates an answer strictly from the retrieved context using Google Gemini.
3. **Critic Agent**: Verifies whether the answer is supported by the paper and triggers a revision if unsupported claims are found.

---

## Architecture Workflow

```
Research Paper (PDF)
        ↓
PyMuPDF Text Extraction
        ↓
Word-Boundary Chunking
        ↓
Gemini Embeddings
        ↓
FAISS Vector Store
        ↓
User Question
        ↓
Retriever Agent (Top-k chunks with page numbers)
        ↓
Research Analyst Agent (Grounded synthesis)
        ↓
Critic Agent (Factual verification)
        ↓
Final Answer + Sources (Streamlit UI)
```

---

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: Clean, interactive web interface.
- **Google Gemini API**: `gemini-flash-latest` for analysis and `gemini-embedding-2` for embeddings.
- **FAISS**: Local vector database for fast similarity search.
- **PyMuPDF (`fitz`)**: Fast PDF text extraction preserving page provenance.
- **python-dotenv**: Environment variable management.

---

## Project Structure

```
research-paper-analyzer/
│
├── app.py                  # Streamlit user interface
├── config.py               # Configuration constants and models
├── pipeline.py             # Agentic RAG coordinator (Retriever -> Analyst -> Critic)
├── requirements.txt        # Project dependencies
├── .env.example            # Environment template
├── .gitignore              # Files ignored by git
├── LICENSE                 # MIT License
├── README.md               # Project documentation
│
├── agents/
│   ├── retriever_agent.py  # Retrieves relevant paper chunks
│   ├── analyst_agent.py    # Synthesizes answers and handles revisions
│   └── critic_agent.py     # Verifies factual grounding
│
├── rag/
│   ├── pdf_loader.py       # Extracts and normalizes text from PDF
│   ├── text_splitter.py    # Chunks text on clean word boundaries
│   ├── embeddings.py       # Multi-part batch embedding with Gemini
│   ├── vector_store.py     # FAISS vector indexing
│   └── retriever.py        # Semantic similarity search
│
├── utils/
│   └── prompts.py          # Prompt templates for agents
│
└── data/
    └── uploads/            # Temporary storage for uploaded papers
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/7091arvind-Git/research-paper-analyzer.git
cd research-paper-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file from the example:
```bash
copy .env.example .env
```
Open `.env` and add your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment / Streamlit Secrets

When deploying to **Streamlit Community Cloud**:
1. In your Streamlit Cloud dashboard, go to your app's **Settings** -> **Secrets**.
2. Add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
3. Save and deploy. Users visiting the deployed application do not need to enter an API key; the application reads it securely from Streamlit Secrets.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
