from dotenv import load_dotenv
import os

load_dotenv()

pine_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")