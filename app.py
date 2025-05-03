from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load DistilBERT sentiment model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_fake_news(content):
    # Tokenize input
    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Get highest probability class (0 = negative, 1 = positive)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence_score = torch.max(probs).item()

    # Interpret the prediction
    # 0 = Negative (fake), 1 = Positive (true)
    if predicted_class == 1:
        label = "True"
    else:
        label = "False"

    return label, round(confidence_score * 100, 2)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    error_message = None

    if request.method == "POST":
        content = request.form.get("news_content", "").strip()
        
        if not content:
            error_message = "Please enter news content."
        else:
            result, confidence = predict_fake_news(content)

    return render_template("index.html", result=result, confidence=confidence, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
 