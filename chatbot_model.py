import nltk
import json
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Pobranie tokenizera
nltk.download('punkt')

# Załaduj dane
with open('data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Przygotowanie danych
patterns = []
responses = []
intents_list = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        intents_list.append(intent['intent'])

# Tokenizacja
words = nltk.word_tokenize(' '.join(patterns))
words = list(set(words))

# Funkcja czyszcząca
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [word.lower() for word in sentence_words if word.lower() in words]

# TF-IDF
vectorizer = TfidfVectorizer(tokenizer=clean_up_sentence, stop_words=None)
X = vectorizer.fit_transform(patterns)

# Etykiety
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents_list)

# Model
model = SVC(kernel='linear', probability=False)
model.fit(X, y)

# Licznik błędów (importowany w app.py)
fail_count = 0

# Przewidywanie z confidence
def predict_intent_with_confidence(text):
    vec = vectorizer.transform([text])
    confidence = vec.max()  # poziom dopasowania TF-IDF

    if confidence < 0.2:  # PRÓG — można regulować
        return None

    prediction = model.predict(vec)
    intent = label_encoder.inverse_transform(prediction)
    return intent[0]


def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['intent'] == intent:
            return random.choice(intent_data['responses'])
    return "Przepraszam, nie rozumiem pytania."
