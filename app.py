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
    response = jsonify({'response': response_text})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response



if __name__ == '__main__':
    app.run(debug=True)
