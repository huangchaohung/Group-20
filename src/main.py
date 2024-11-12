from flask import Flask, render_template, request, jsonify, send_file
from recommendation_system import get_recommendation, feature_descriptions, product_features
from roi import calculate_and_plot_all_rois
from dynamic_email import dynamic_email_modifier
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/roi_analysis')
def roi_analysis():
    # Define paths for the overall model and dataset
    overall_model_path = os.path.join(os.path.dirname(__file__), "roi_models", "random_forest_model_overall.pkl")
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/digital_marketing_campaign_dataset.csv")

    # Generate the ROI plot
    plot_url = calculate_and_plot_all_rois(overall_model_path, dataset_path)

    return render_template('roi.html', plot_url=plot_url)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_choice = data.get("product_choice")
    user_data = data.get("user_data", {})
    response = get_recommendation(product_choice, user_data)
    return jsonify(response)

@app.route('/feature_descriptions', methods=['GET'])
def get_feature_descriptions():
    return jsonify(feature_descriptions)

@app.route('/product_features', methods=['GET'])
def get_product_features():
    return jsonify(product_features)


@app.route('/dynamic_email', methods=['GET', 'POST'])
def dynamic_email():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.csv'):
            recent_email_result = pd.read_csv(file)
            differences, similarities, download_path = dynamic_email_modifier(recent_email_result)
            return render_template(
                'dynamic_email_result.html',
                differences=differences,
                similarities=similarities,
                download_path=download_path
            )
    return render_template('dynamic_email.html')

@app.route('/download_features')
def download_features():
    download_path = request.args.get('download_path')
    if download_path and os.path.exists(download_path):
        return send_file(download_path, as_attachment=True)
    return "File not found.", 404

if __name__ == '__main__':
    app.run(debug=True)