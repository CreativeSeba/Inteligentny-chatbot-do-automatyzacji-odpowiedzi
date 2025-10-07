import nltk
import json
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Załaduj dane
nltk.download('punkt')

with open('data/intents.json') as file:
    intents = json.load(file)

# Przygotuj dane
patterns = []
responses = []
intents_list = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        intents_list.append(intent['intent'])

# Tokenizacja
nltk.download('punkt')
words = nltk.word_tokenize(' '.join(patterns))
words = list(set(words))

# Znakowanie słów
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [word.lower() for word in sentence_words if word in words]

# Konwertowanie odpowiedzi do wektorów
vectorizer = TfidfVectorizer(tokenizer=clean_up_sentence, stop_words=None)
X = vectorizer.fit_transform(patterns)

# Kodowanie etykiet (intencji)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents_list)

# Trening modelu
model = SVC(kernel='linear')
model.fit(X, y)

# Funkcja do przewidywania intencji
def predict_intent(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    intent = label_encoder.inverse_transform(prediction)
    return intent[0]

# Funkcja zwracająca odpowiedź
def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['intent'] == intent:
            return random.choice(intent_data['responses'])
    return "Przepraszam, nie rozumiem pytania."
