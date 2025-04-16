from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_doc(file_path):
    file_loader = PyPDFLoader(file_path)
    document = file_loader.load()
    print(f"Loaded document with {len(document)} pages")
    return document


def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc