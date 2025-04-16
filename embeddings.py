from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()