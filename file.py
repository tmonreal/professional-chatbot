# %%
import os
import time
from dotenv import load_dotenv
from utils import read_doc, chunk_data
from embeddings import NormalizedSentenceTransformerEmbedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# %%
# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# %%
# Read PDF
file_path = r'docs/Trinidad_Monreal_Resume.pdf'
total = read_doc(file_path)

# %%
# Split into chunks
chunked_docs = chunk_data(docs=total, chunk_size=500, chunk_overlap=100)

# %%
# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'tmonreal'

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Index {index_name} deleted")

pc.create_index(
    name=index_name,
    dimension=512,
    metric='cosine',
    spec=spec
)
print(f"Index {index_name} created")

# %%
# Load embedding model
embedding_model = NormalizedSentenceTransformerEmbedding("jinaai/jina-embeddings-v2-small-en")

print(f"\n🧪 Number of chunks: {len(chunked_docs)}")
for i in range(min(3, len(chunked_docs))):
    print(f"\nChunk {i+1} preview:\n{chunked_docs[i].page_content[:200]}")

# Create and print first vector
first_vec = embedding_model.embed_query(chunked_docs[0].page_content)
print(f"\n🔎 First vector length: {len(first_vec)}")
print(f"First 5 values: {first_vec[:5]}")

# Check full upsert payload
print(f"\n📦 Sample vector for upsert:\n{{\
    'id': 'doc-0',\
    'values': {first_vec[:5]}...,\
    'metadata': {{'text': chunked_docs[0].page_content[:60]}}}}")

# %%
# MANUAL upsert of document vectors
namespace = "espacio"
ids = [f"doc-{i}" for i in range(len(chunked_docs))]
vectors = [
    {
        "id": ids[i],
        "values": embedding_model.embed_query(doc.page_content),
        "metadata": {"text": doc.page_content}
    }
    for i, doc in enumerate(chunked_docs)
]

index = pc.Index(index_name)
index.upsert(vectors=vectors, namespace=namespace)

# Confirm upsert worked
stats = index.describe_index_stats()
print("💡 Upsert complete. Vector count:", stats['total_vector_count'])

# %%
# Set up retriever from Pinecone
vectorstore = PineconeVectorStore(index_name=index_name,
                                  embedding=embedding_model,
                                  namespace=namespace)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# %%
# Test retrieval
query = "in which companies did trinidad used to work"
print(f"\nQuery: {query}")

docs = retriever.invoke(query)
print(f"\nRetrieved {len(docs)} document(s):")
for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---\n{doc.page_content[:400]}...\n")

# Optional: test raw vector similarity manually
query_vector = embedding_model.embed_query(query)
results = index.query(
    vector=query_vector,
    namespace=namespace,
    top_k=3,
    include_metadata=True
)
print(f"\nPinecone returned {len(results['matches'])} matches")
for i, match in enumerate(results['matches']):
    print(f"\nMatch {i+1} (score: {match['score']:.4f})")
    print(match['metadata']['text'][:400])

# %%
# Set up RAG pipeline with Groq
groq_model = ChatGroq(api_key=GROQ_API_KEY, 
                      model="meta-llama/llama-4-scout-17b-16e-instruct")

qa = RetrievalQA.from_chain_type(
    llm=groq_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Final QA call
result = qa.invoke(query)
print("\nAnswer:", result['result'])

for i, doc in enumerate(result['source_documents']):
    print(f"\n🔎 Source {i+1}:\n{doc.page_content[:400]}")



"""
#nlp = spacy.load("en_core_web_sm")  # You can use "es_core_news_sm" for Spanish CVs
nlp = spacy.load("es_core_news_sm")

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    # Define a basic regular expression to detect potential section headers (capitalized)
    section_headers = re.compile(r"\b([A-ZÁÉÍÓÚ][A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)\b")

    # Initialize the sentence splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ",", ".", "!", "?"]  # Sentence boundaries
    )

    chunked_docs = []
    
    for doc in docs:
        content = doc.page_content

        # Split content by sentences using SpaCy
        spacy_doc = nlp(content)
        sentences = [sent.text.strip() for sent in spacy_doc.sents]

        section_name = None
        current_section = []

        # Process sentences and detect section changes
        for sentence in sentences:
            # Check if the sentence matches a section header
            potential_header = section_headers.match(sentence)
            if potential_header:
                # When we find a new section header, save the previous section if it exists
                if current_section:
                    chunked_docs.append(f"{section_name} {''.join(current_section)}")
                section_name = potential_header.group(0)  # Capture section header
                current_section = [sentence]  # Start new section
            else:
                # Append sentence to the current section
                current_section.append(sentence)

        # Add the last section if there's any content left
        if current_section:
            chunked_docs.append(f"{section_name} {''.join(current_section)}")

    return chunked_docs
"""