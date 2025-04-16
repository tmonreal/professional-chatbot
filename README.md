# 💬 Trinidad Monreal's Professional ChatBot — RAG-based CV Assistant

This is a test project implementing **Retrieval-Augmented Generation (RAG)** to answer questions about **Trinidad Monreal's professional background**. It combines information from her **CV** and **LinkedIn profile**, using LangChain, Pinecone, and Groq's **LLaMA 4 17B Scout**.

---

## 🧠 How It Works

1. CV and LinkedIn PDF are parsed and chunked.
2. Embeddings are generated using [`jinaai/jina-embeddings-v2-small-en`](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) via Hugging Face.
3. Vectors are stored in Pinecone.
4. A user query is embedded and matched with the most relevant chunks using cosine similarity.
5. The matched context and query are sent to Groq’s LLaMA 4 model for a final answer.

### 📊 Architecture Diagram

![RAG Diagram](static/diagram.jpg)

---

## 🚀 Running the App Locally

### 1. Clone the repo and set up your environment

```bash
git clone https://github.com/tmonreal/professional-chatbot.git
cd professional-chatbot
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
