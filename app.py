# application.py

from flask import Flask, render_template, request, jsonify
from llm_engine import load_qa_chain

app = Flask(__name__)

# Load the LLM + Retriever chain once on startup
qa = load_qa_chain()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_message = request.form.get("msg")

    if not user_message:
        return jsonify({"answer": "I didn't get your message."})

    try:
        result = qa.invoke({"query": user_message})
        return jsonify({"answer": result["result"]})
    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)