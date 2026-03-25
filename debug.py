# debug.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="indigo_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_vectorstore",
)

# ── Check 1: How many chunks are stored? ──
print(f"Total chunks in vector store: {vector_store._collection.count()}")

# ── Check 2: Search with different queries ──
queries = [
    "IndiGo founded",
    "IndiGo established",
    "IndiGo history",
    "when was IndiGo started",
    "2006",
    "InterGlobe Aviation",
]

for q in queries:
    docs = vector_store.similarity_search(q, k=2)
    print(f"\n--- Query: '{q}' ---")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] Source: {doc.metadata.get('source')}")
        print(f"       Preview: {doc.page_content[:300]}")
        print()
# Add this at the bottom of debug.py
print("\n\n=== SIMULATING STREAMLIT QUERY ===")
test_q = "when was indigo founded"
docs = vector_store.similarity_search(test_q, k=3)
print(f"Query: '{test_q}'")
print(f"Chunks found: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n[{i+1}] Source: {doc.metadata.get('source')}")
    print(f"     Full content:\n{doc.page_content[:500]}")