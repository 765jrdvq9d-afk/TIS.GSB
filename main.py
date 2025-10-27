from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load AI model (Flan-T5 Small)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

def ask_bot(question):
    # Read all notes dynamically every time
    with open("notes.txt", "r", encoding="utf-8") as f:
        all_notes = f.read()
    
    prompt = f"Use the following notes to answer the question:\n{all_notes}\nQuestion: {question}\nAnswer:"
    answer = qa_model(prompt, max_length=200)[0]["generated_text"]
    return answer[len(prompt):].strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    answer = ask_bot(question)
    return {"answer": answer}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
