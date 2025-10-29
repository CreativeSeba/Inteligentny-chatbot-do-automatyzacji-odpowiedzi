from flask import Flask, render_template, request, jsonify
from chatbot_model import predict_intent, get_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



from flask import jsonify

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    intent = predict_intent(user_input)
    response_text = get_response(intent)
    return jsonify({'response': response_text})


if __name__ == '__main__':
    app.run(debug=True)
