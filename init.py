import os
import time
from dotenv import load_dotenv
from utils import read_doc, chunk_data
from embeddings import SentenceTransformerEmbedding
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# Load environment variables from .env file and store them in variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up the LLM Groq model 
groq_model = ChatGroq(api_key=GROQ_API_KEY, 
                      model="meta-llama/llama-4-scout-17b-16e-instruct", 
                      temperature=0,
                      streaming=True)

# Read my CV
file_path = r'docs/Trinidad_Monreal_Resume.pdf'
total = read_doc(file_path)

# Split the document into chunks
chunked_docs = chunk_data(docs=total, chunk_size=300, chunk_overlap=50)
type(chunked_docs)

# Connect to Pinecone DB and manage index
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'tmonreal'

# If the index exists, delete it
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print("Index {} deleted".format(index_name))

# Create a new index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("Index created with the name: {}".format(index_name))
    pc.create_index(
        index_name,
        dimension=512,  # dimensionality of text-embedding jina-embeddings-v2-small-en
        metric='cosine',
        spec=spec
    )
else:
    print("Index with the name {} already exists".format(index_name))

# Load embedding model
raw_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
embedding_model = SentenceTransformerEmbedding(raw_model)

# Create and upsert embeddings into Pinecone
namespace = "espacio"
docsearch = PineconeVectorStore.from_documents(documents=chunked_docs, 
                                               index_name=index_name,
                                               embedding=embedding_model,
                                               namespace=namespace
                                               )

index = pc.Index(index_name)
stats = index.describe_index_stats()

for _ in range(5):  
    stats = index.describe_index_stats()
    count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    if count >= len(chunked_docs):
        print(f"Pinecone confirmed {count} vectors in namespace '{namespace}'")
        break
    print(f"⏳ Still waiting for Pinecone to reflect changes in {index_name} index")
    time.sleep(2)
else:
    print("Vectors still not found in Pinecone after retrying.")

# Retrieve and search into the created Pinecone index
vectorstore = PineconeVectorStore(index_name=index_name,
                                  embedding=embedding_model,
                                  namespace=namespace
                                  )
retriever = vectorstore.as_retriever()

query = "What university did Trinidad attend? What did she study?"
print(f"Query: {query}")
vectorstore.similarity_search(query, k=3)

qa = RetrievalQA.from_chain_type(llm=groq_model,
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever()
                                 )

result = qa.invoke(query)
print("Answer:", result['result'])