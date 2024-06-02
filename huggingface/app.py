from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)



pipe = pipeline("text-classification", model="mapsoriano/roberta-tagalog-base-philippine-elections-2016-2022-hate-speech")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    result = pipe(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
