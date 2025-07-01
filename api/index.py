from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, pdfplumber
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

load_dotenv()
app = FastAPI()

PDF_DIR = "data"

# ✅ Serve index.html from root (since it's at root, not in /static)
@app.get("/", include_in_schema=False)
async def serve_home():
    return FileResponse("index.html")

@app.get("/api")
async def list_pdfs():
    files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    return {"files": files}

@app.post("/api")
async def ask(question: str = Form(...), filename: str = Form(...)):
    try:
        path = os.path.join(PDF_DIR, filename)
        with pdfplumber.open(path) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)

        if not text.strip():
            return JSONResponse(content={"error": "No text found in PDF"}, status_code=400)

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        docs = vectorstore.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatCohere(model="command-r-plus", temperature=0.3)
        prompt = PromptTemplate.from_template(
            "Answer the following question based on the context:\n\n{context}\n\nQuestion: {question}"
        )

        chain = (
            RunnableMap({"context": lambda x: context, "question": lambda x: x["question"]})
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke({"question": question})
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
