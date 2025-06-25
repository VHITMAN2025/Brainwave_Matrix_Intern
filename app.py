from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(token.lower() for token in str(text).split() if token not in stopwords.words('english'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        cleaned = clean_text(text)
        X_new = vectorizer.transform([cleaned])
        pred = model.predict(X_new)
        prediction = 'REAL' if pred[0] == 1 else 'FAKE'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)