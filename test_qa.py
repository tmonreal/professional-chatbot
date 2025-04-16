from llm_engine import load_qa_chain

qa = load_qa_chain()

query = "What programming languages does Trinidad know?"
result = qa.invoke({"query": query})

print("Query:", query)
print("Answer:", result["result"])