"""
Use this script to load the querying interface for my Flask app or CLI.
"""
import os
from dotenv import load_dotenv
from embeddings import SentenceTransformerEmbedding
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def load_qa_chain():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    index_name = 'tmonreal'
    namespace = 'espacio'

    # Load retriever
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embedding_model = SentenceTransformerEmbedding(
        SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
    )
    retriever = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
        namespace=namespace
    ).as_retriever()

    # Load LLM
    groq_model = ChatGroq(api_key=GROQ_API_KEY,
                          model="meta-llama/llama-4-scout-17b-16e-instruct",
                          temperature=0,
                          streaming=False)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are analyzing a CV belonging to Trinidad Monreal (female). Use the context below to answer the question.  
        If you don't know the answer, say 'I don't know'. 
        Use three sentences maximum and keep the answer concise.

        Context:
        {context}

        Question: {question}
        Helpful Answer:
        """,
        input_variables=["context", "question"]
    )

    # Build QA chain
    qa = RetrievalQA.from_chain_type(
        llm=groq_model,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa