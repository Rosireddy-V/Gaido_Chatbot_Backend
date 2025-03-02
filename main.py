from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys securely (set these as environment variables)
groq_api_key = os.environ.get("GROQ_API_KEY")
mistralai_api_key = os.environ.get("MISTRALAI_API_KEY")
# Load LLM & Embeddings
llm = init_chat_model("llama3-70b-8192", model_provider="groq")
embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=mistralai_api_key)
vector_store = InMemoryVectorStore(embeddings)

# Load and split document
file_path = "Sample HI Policy.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Index documents in vector store
vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define request model for API
class QueryRequest(BaseModel):
    question: str

# Define retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Define generation step
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve","generate")
graph = graph_builder.compile()

# API endpoint for question answering
@app.post("/query")
async def query(request: QueryRequest):
    try:
        response = graph.invoke({"question": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI RAG Chatbot is running!"}
