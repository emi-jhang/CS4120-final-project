from flask import Flask, request, jsonify
from flask_cors import CORS
from dummy_script import simplify_text

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/')
def home():
    return "Flask server is running!"


@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.json  # Get JSON data from the POST request
    input_text = data.get('text', '')  # Extract 'text' field

    if not input_text.strip():  # Check if text is empty
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Simplify the text using your function
        simplified_text = simplify_text(input_text)

        return jsonify({'simplified_text': simplified_text})
    except Exception as e:
        # Handle any unexpected errors
        # return jsonify({'error': str(e)}), 500
        return jsonify({'Hello', 500})

if __name__ == '__main__':
    app.run(debug=True)
