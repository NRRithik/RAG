import os
import time
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    def __init__(
        self,
        doc_path: str = "sample_document.txt",
        api_key: str | None = None
    ):
        # 1. API Key Setup
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # 2. Document Loading & Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        loader = TextLoader(doc_path, encoding="utf-8")
        docs = loader.load()
        self.chunks = splitter.split_documents(docs)

        # 3. Vector Database (ChromaDB)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        unique_col_name = f"col_{uuid.uuid4().hex[:10]}"
        vector_store = Chroma.from_documents(self.chunks, embeddings, collection_name=unique_col_name)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # 4. Prompt Template
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant answering questions about the person described in the profile context below.

        Profile Context:
        {context}

        Question: {question}

        Guidelines:
        - Answer using ONLY the provided profile context.
        - You can answer naturally in first person ("I am...", "I work at...") or third person depending on how the question is phrased.
        - If the question cannot be answered from the provided context, politely say: "I don't have that information in my profile."
        - Be concise, accurate, and professional. Do not invent or assume facts not found in the context.
        """)

        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        # 5. LCEL Pipeline
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Invokes RAG with automatic retries for temporary Google server spikes."""
        for attempt in range(4):
            try:
                return self.chain.invoke(question)
            except Exception as e:
                if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 3:
                    time.sleep(3)
                    continue
                raise e
