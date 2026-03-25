# ✈️ IndiGo Airline RAG Chatbot

> An AI-powered chat assistant for IndiGo Airlines — built with Retrieval-Augmented Generation (RAG), LangChain, ChromaDB, and GPT-4o-mini. Ask anything about IndiGo: history, incidents, flight schedules, policies, and more.

🔗 **Live Demo**: [indigoairlinesragchatbot-7ecjijrfuu92rmcrepzs87.streamlit.app](https://indigoairlinesragchatbot-7ecjijrfuu92rmcrepzs87.streamlit.app)

---


## 🧠 How It Works

```
User Question
     │
     ▼
Streamlit UI (app.py)
     │
     ▼
Similarity Search → ChromaDB Vector Store
     │
     ▼
Top-3 Relevant Chunks (PDF / CSV data)
     │
     ▼
GPT-4o-mini (via LangChain) generates answer
     │
     ▼
Answer displayed in chat
```

The app uses **RAG (Retrieval-Augmented Generation)**:
1. Documents are chunked and embedded using OpenAI's `text-embedding-3-small`
2. Embeddings are stored in a local **ChromaDB** vector store
3. At query time, the top-3 most relevant chunks are retrieved
4. A prompt with the retrieved context is sent to **GPT-4o-mini**
5. The LLM answers strictly based on the provided context

---

## 📁 Project Structure

```
Indigo_airlines_RAG_chatbot/
│
├── app.py                  # Streamlit frontend
├── main.py                 # RAG query logic (get_response)
├── ingest.py               # One-time data ingestion pipeline
│
├── indigo_data/            # Source documents
│   ├── IndiGo-Factsheet.pdf
│   ├── indigo_incidents_data.csv
│   ├── flight_sch_W24_...pdf
│   └── IGAL - WhistleBlower Policy.pdf
│
├── chroma_vectorstore/     # Persisted vector store (auto-generated)
├── requirements.txt
├── .env                    # Local secrets (never commit this!)
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/urmikanrar2003-uk/Indigo_airlines_RAG_chatbot.git
cd Indigo_airlines_RAG_chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI API key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

> 🔑 Get your API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 5. Ingest your documents (first time only)

Place your PDF and CSV files inside the `indigo_data/` folder, then run:

```bash
python ingest.py
```

This will:
- Load all PDFs and CSVs from `indigo_data/`
- Split them into chunks (size: 1000, overlap: 200)
- Embed them using `text-embedding-3-small`
- Save the vector store to `./chroma_vectorstore/`

> ⚠️ Only run `ingest.py` once (or when your data changes). Do **not** run it on every app start.

### 6. Run the app locally

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🌐 Deployment (Streamlit Cloud)

This app is deployed on **Streamlit Cloud** with the vector store committed to the repository.

### Steps to deploy your own:

1. Push your code (including `chroma_vectorstore/`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository and set main file to `app.py`
4. Under **Advanced settings**:
   - Set Python version to `3.11`
   - Add your secret:
     ```toml
     OPENAI_API_KEY = "sk-your-key-here"
     ```
5. Click **Deploy!**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector Store | ChromaDB (local) |
| RAG Framework | LangChain |
| PDF Parsing | PyPDF, pdfplumber |
| CSV Parsing | LangChain CSVLoader |
| Deployment | Streamlit Cloud |

---

## 📄 Data Sources

The chatbot answers questions based on the following IndiGo Airlines documents:

- **IndiGo Factsheet** — founding, history, fleet, key stats
- **Incidents CSV** — historical flight incident records
- **Flight Schedule PDF** — Winter 2024 schedule (W24)
- **Whistleblower Policy** — corporate governance document

---

## ⚙️ Configuration

You can tune the RAG pipeline in `ingest.py` and `main.py`:

```python
# Chunk size (ingest.py)
RecursiveCharacterTextSplitter(
    chunk_size=1000,    # ← increase for more context per chunk
    chunk_overlap=200   # ← increase to reduce information loss at boundaries
)

# Number of chunks retrieved per query (main.py)
vector_store.similarity_search(question, k=3)  # ← increase k for more context
```

---

## 🔒 Security Notes

- Never commit your `.env` file — it's listed in `.gitignore`
- Rotate your OpenAI API key immediately if accidentally pushed to GitHub
- On Streamlit Cloud, secrets are managed via the **Advanced Settings** panel

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📝 License

This project is for educational purposes.

---

<div align="center">
  Built with ❤️ using LangChain, ChromaDB, and Streamlit
</div>
