from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from gtts import gTTS
import os
from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json', encoding='utf8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)

@app.route('/get', methods=['POST'])
def get_bot_response():
    userText = request.form['msg']
    # Xử lý tin nhắn từ người dùng và tạo phản hồi của chatbot
    bot_response = chatbot_response(userText)
    return bot_response

@app.route('/tts', methods=['POST'])
def tts():
    text = request.form['text']
    tts = gTTS(text, lang='vi')
    tts.save("response.mp3")
    return send_file("response.mp3", as_attachment=True)

@app.route('/favicon')
def favicon():
    return send_from_directory(app.root_path, './templates/favicon.png', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

@app.route('/intents', methods=['GET'])
def get_intents():
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents_data = f.read()
    return jsonify(json.loads(intents_data))

def get_Chat_response(msg):
    res = chatbot_response(msg)
    return res

if __name__ == '__main__':
    app.run()
