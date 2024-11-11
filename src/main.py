from flask import Flask, request, jsonify, render_template
from recommendation_system import get_recommendation, feature_descriptions, product_features

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves index.html from the templates folder

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_choice = data.get("product_choice")
    user_data = data.get("user_data", {})

    response = get_recommendation(product_choice, user_data)
    if "error" in response:
        return jsonify(response), 400
    return jsonify(response)

@app.route('/feature_descriptions', methods=['GET'])
def get_feature_descriptions():
    return jsonify(feature_descriptions)

@app.route('/product_features', methods=['GET'])
def get_product_features():
    return jsonify(product_features)

if __name__ == '__main__':
    app.run(debug=True)
