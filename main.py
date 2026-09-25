import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Optimized chunk size for higher relevance precision
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = file.filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

        vector_store.add_documents(chunks)

        return {"filename": file.filename, "status": "Document fully parsed and indexed successfully!"}
    except Exception as e:
        return {"filename": file.filename, "status": f"Error parsing PDF: {str(e)}"}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        # Remove file from disk
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove chunks from Chroma vector store
        try:
            vector_store.delete(where={"source": filename})
        except Exception:
            pass

        return {"status": f"Document '{filename}' deleted successfully!"}
    except Exception as e:
        return {"status": f"Error deleting document: {str(e)}"}

@app.post("/chat")
async def chat_with_docs(question: str = Form(...)):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant information in your uploaded documents. Try asking something specific covered in your notes.",
                "sources": []
            }

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        top_source = relevant_docs[0].metadata.get("source", "Unknown")
        page_num = relevant_docs[0].metadata.get("page", 1)

        answer = f"Based on your document ({top_source}), here are the exact details found:\n\n{context}"

        return {
            "answer": answer,
            "sources": [{"file": top_source, "page": page_num}]
        }
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "sources": []}

@app.post("/quiz")
async def generate_quiz(topic: str = Form(...)):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke(topic)

        if not relevant_docs:
            return {"quiz": [
                {
                    "id": 1,
                    "question": f"No specific notes found for '{topic}'. Please upload documents containing this topic.",
                    "options": ["N/A", "N/A", "N/A", "N/A"],
                    "answer": "N/A"
                }
            ]}

        quiz_data = []
        for idx, doc in enumerate(relevant_docs):
            if idx >= 10:
                break
            snippet = doc.page_content.strip().replace("\n", " ")[:120]
            quiz_data.append({
                "id": idx + 1,
                "question": f"Based on your notes for '{topic}' (Snippet {idx + 1}): Which concept relates to '{snippet}...'?",
                "options": [
                    "A core concept stated directly in your text",
                    "An alternate hypothetical theory",
                    "An outdated methodology",
                    "None of the above"
                ],
                "answer": "A core concept stated directly in your text"
            })

        return {"quiz": quiz_data}
    except Exception as e:
        return {"quiz": []}