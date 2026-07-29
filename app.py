from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

def load_text_with_encoding(filepath):
    encodings = ["utf-8", "utf-16", "cp1252"]
    for enc in encodings:
        try:
            loader = TextLoader(filepath, encoding=enc)
            return loader.load()
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not decode {filepath} with any known encoding")

# Load the two PDFs
taskbook_loader = PyPDFLoader("data/AI-ML-TasksBook - Copy.pdf")
guidelines_loader = PyPDFLoader("data/Guidelines - Copy.pdf")
pdf_docs = taskbook_loader.load() + guidelines_loader.load()

# Load the 3 READMEs with encoding fallback
readme_files = ["README.md", "README_2.md", "README_3.md"]
readme_docs = []
for filename in readme_files:
    readme_docs.extend(load_text_with_encoding(f"data/{filename}"))

# Combine everything
all_documents = pdf_docs + readme_docs

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(all_documents)

print(f"Loaded {len(all_documents)} document(s), split into {len(chunks)} chunks")

# Create the embedding model (downloads automatically on first run)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed all your chunks and store them in a FAISS index
vector_store = FAISS.from_documents(chunks, embedding_model)

# Save the index to disk so you don't have to re-embed every time you run the app
vector_store.save_local("faiss_index")
print("Vector store created and saved successfully")

load_dotenv()  # reads your .env file

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

query = "LangChain LlamaIndex FAISS ChromaDB Streamlit RAG chatbot"
results = vector_store.similarity_search(query, k=3)

print("--- Retrieved chunks ---")
for i, doc in enumerate(results):
    print(f"Chunk {i+1}: {doc.page_content[:150]}...")
    print()

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't have information about that in the provided documents."

Context:
{context}

Question: {query}

Answer:"""

response = llm.invoke(prompt)
print(response.content)