from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from uuid import uuid4
import pdfplumber
import os

# ─────────────────────────────────────────
# 1. PDF LOADERS
# ─────────────────────────────────────────

def load_pdf_with_tables(pdf_path):
    """Use pdfplumber for PDFs with tables (e.g. flight schedules)"""
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            table_text = ""
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                headers = [cell if cell else "" for cell in table[0]]
                for row in table[1:]:
                    row_clean = [cell if cell else "" for cell in row]
                    row_str = " | ".join(
                        f"{headers[i]}: {row_clean[i]}"
                        for i in range(min(len(headers), len(row_clean)))
                    )
                    table_text += row_str + "\n"

            full_text = text + "\n" + table_text
            if full_text.strip():
                documents.append(Document(
                    page_content=full_text,
                    metadata={"source": pdf_path, "page": page_num}
                ))
    return documents


# ─────────────────────────────────────────
# 2. CSV LOADER
# ─────────────────────────────────────────

def load_csv_as_documents(csv_path):
    """Load CSV — each row becomes one Document"""
    loader = CSVLoader(
        file_path=csv_path,
        encoding="utf-8-sig",       # handles the BOM from utf-8-sig saved files
        csv_args={"delimiter": ","}
    )
    docs = loader.load()
    print(f"[CSVLoader] Loaded {len(docs)} rows from {os.path.basename(csv_path)}")
    return docs


# ─────────────────────────────────────────
# 3. LOAD ALL FILES FROM FOLDER
# ─────────────────────────────────────────

def load_all_files(directory):
    """Route each file to the right loader based on type/filename"""
    all_docs = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        # ── CSVs ──
        if filename.endswith(".csv"):
            print(f"[CSVLoader]    Loading: {filename}")
            all_docs.extend(load_csv_as_documents(path))

        # ── PDFs ──
        elif filename.endswith(".pdf"):
            if "schedule" in filename.lower():
                print(f"[pdfplumber]   Loading: {filename}")
                all_docs.extend(load_pdf_with_tables(path))
            else:
                print(f"[PyPDFLoader]  Loading: {filename}")
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())

        else:
            print(f"[Skipped]      {filename}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs


# ─────────────────────────────────────────
# 4. EMBEDDINGS & VECTOR STORE
# ─────────────────────────────────────────

embeddings = OllamaEmbeddings(
    model="bge-large:335m",
    base_url="http://localhost:11434"
)

vector_store = Chroma(
    collection_name="indigo_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_vectorstore",
)

# ─────────────────────────────────────────
# 5. LOAD + CHUNK + STORE (First time only)
# ─────────────────────────────────────────

# Uncomment only on first run — comment out after
# ------------------------------------------------
# documents = load_all_files("indigo_data")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# texts = text_splitter.split_documents(documents)
# print(f"Total chunks after splitting: {len(texts)}")

# uuids = [str(uuid4()) for _ in range(len(texts))]
# vector_store.add_documents(documents=texts, ids=uuids)
# print("Vector store created and saved ✅")
# #------------------------------------------------


# ─────────────────────────────────────────
# 6. LLM
# ─────────────────────────────────────────

llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
    top_p=0.1,
)


# ─────────────────────────────────────────
# 7. RAG — GET RESPONSE
# ─────────────────────────────────────────

def get_response(question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')} | Page/Row: {d.metadata.get('page', d.metadata.get('row', '?'))}]\n{d.page_content}"
        for d in docs
    ])

    prompt =  f"""You are a helpful assistant for IndiGo airline queries.

Extract ALL incidents related to the location mentioned in the question.

If multiple incidents exist, list them clearly.

Context:
{context}

Question:
{question}

Answer:"""

    return llm.invoke(prompt)


# ─────────────────────────────────────────
# 8. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    question = "What incidents happened at Gorakhpur?"
    answer = get_response(question)
    print(f"Q: {question}")
    print(f"A: {answer}")


## Key things to know:

# **CSVLoader behavior** — each row becomes a separate Document like this:
# ```
# Title: Indigo A21N at Gorakhpur on Jan 11th 2026, bird strike
# Aircraft_Type: A21N
# Location: Gorakhpur
# Cause: bird strike
# Incident_Type: Incident
# ...