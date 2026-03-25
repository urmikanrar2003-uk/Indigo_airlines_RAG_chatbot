# ingest.py — run once to build the vector store

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import pdfplumber
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# LOADER FUNCTIONS (copied from your main.py)
# ─────────────────────────────────────────

def load_pdf_with_tables(pdf_path):
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


def load_csv_as_documents(csv_path):
    loader = CSVLoader(
        file_path=csv_path,
        encoding="utf-8-sig",
        csv_args={"delimiter": ","}
    )
    docs = loader.load()
    print(f"[CSVLoader] Loaded {len(docs)} rows from {os.path.basename(csv_path)}")
    return docs


def load_all_files(directory):
    all_docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".csv"):
            print(f"[CSVLoader]    Loading: {filename}")
            all_docs.extend(load_csv_as_documents(path))
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
# INGEST
# ─────────────────────────────────────────

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="indigo_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_vectorstore",
)

documents = load_all_files("indigo_data")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)
print(f"Total chunks after splitting: {len(texts)}")

for doc in texts:
    doc.page_content = ' '.join(doc.page_content.split())

uuids = [str(uuid4()) for _ in range(len(texts))]
vector_store.add_documents(documents=texts, ids=uuids)
print("Vector store created and saved ✅")