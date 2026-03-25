# main.py — ONLY query logic, no ingestion on import

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="indigo_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_vectorstore",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_response(question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')} | Page: {d.metadata.get('page', '?')}]\n{' '.join(d.page_content.split())}"
        for d in docs
    ])
    prompt = f"""You are a helpful assistant for IndiGo airline queries.
Use only the context below to answer. If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question:
{question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content