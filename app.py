from flask import Flask, render_template, request, jsonify
from chatbot_model import predict_intent_with_confidence, get_response, fail_count

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global fail_count

    user_input = request.form['message']
    intent = predict_intent_with_confidence(user_input)

    # BRAK DOPASOWANIA
    if intent is None:
        fail_count += 1

        if fail_count >= 3:
            fail_count = 0
            response_text = "Widzę, że masz trudność w znalezieniu informacji.\nSkontaktuj się z nami: www.szkola.pl/kontakt"
        else:
            response_text = "Przepraszam, nie rozumiem pytania."

    # POPRAWNE DOPASOWANIE
    else:
        fail_count = 0
        response_text = get_response(intent)

    response = jsonify({'response': response_text})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


if __name__ == '__main__':
    app.run(debug=True)
