import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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

def chunk_data_sectionwise(docs, chunk_size=500, chunk_overlap=100):
    text = "\n".join([doc.page_content for doc in docs])

    section_titles = [
        "Education",
        "Experience",
        "Courses and Certifications",
        "Projects",
        "Skills",
        "Awards"
    ]

    pattern = r"(?=^(" + "|".join(re.escape(title) for title in section_titles) + r")\s*$)"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    section_docs = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            section_name = matches[i].group(1).strip()
            section_docs.append((section_name, section_text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunked_docs = []
    for section_name, section_text in section_docs:
        sub_chunks = splitter.split_text(section_text)
        for chunk_text in sub_chunks:
            lines = chunk_text.strip().splitlines()
            first_line = lines[0].strip() if lines else ""
            if section_name.lower() in first_line.lower():
                full_text = chunk_text.strip()
            else:
                full_text = f"{section_name}\n{chunk_text.strip()}"
            chunked_docs.append(Document(page_content=full_text, metadata={"section": section_name}))

    return chunked_docs