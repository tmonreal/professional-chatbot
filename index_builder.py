"""
Only run this script when you want to rebuild the Pinecone index (e.g., when CV changes).
"""
import os
import time
from dotenv import load_dotenv
from utils import read_doc, chunk_data_sectionwise
from embeddings import SentenceTransformerEmbedding
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

# Read CV and chunk
cv_path = r'docs/Trinidad_Monreal_Resume.pdf'
linkedin_path = r'docs/LinkedIn_Profile.pdf'

cv_doc = read_doc(cv_path)
linkedin_doc = read_doc(linkedin_path)
total = cv_doc + linkedin_doc
chunked_docs = chunk_data_sectionwise(docs=total, chunk_size=1000, chunk_overlap=100)

for i,chunk in enumerate(chunked_docs):
    print(f"Chunk {i+1}: {chunk.page_content[:50]}...")  # Print first 50 characters of each chunk


# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'tmonreal'
namespace = 'espacio'

# Delete index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Index {index_name} deleted")

# Create index
pc.create_index(
    index_name,
    dimension=512,
    metric='cosine',
    spec=spec
)
print(f"Index created: {index_name}")

# Load embedding model
embedding_model = SentenceTransformerEmbedding(
    SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
)

# Upsert vectors
docsearch = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    index_name=index_name,
    embedding=embedding_model,
    namespace=namespace
)

# Confirm upsert
index = pc.Index(index_name)
for _ in range(5):
    stats = index.describe_index_stats()
    count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    if count >= len(chunked_docs):
        print(f"✅ {count} vectors inserted into namespace '{namespace}'")
        break
    print("⏳ Waiting for vector sync...")
    time.sleep(2)
else:
    print("❌ Timeout: vectors not confirmed.")