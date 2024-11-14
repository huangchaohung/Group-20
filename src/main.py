from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from recommendation_system import get_recommendation, feature_descriptions, product_features
from roi import calculate_and_plot_all_rois
from dynamic_email import dynamic_email_modifier
import pandas as pd
import os
import pickle

app = Flask(__name__)

# Define paths for image and model directories
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "image")
print(IMAGE_FOLDER)
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "segmentation_models")

# Define segmentation sections for the dashboard
segmentation_sections = {
    "Correlation Analysis": {
        "image": "characteristic_corr.png",
        "description": "The features are not correlated at all except for transaction frequency and recency - due to the fact that higher frequency implies a more recent transaction record."
    },
    "Elbow Method": {
        "image": "elbow40.png",
        "description": "The Elbow Method is used to determine the optimal number of clusters for k-means clustering. However, too many customer segments is not realistic in practical banking campaigns."
    },
    "Silhouette Analysis": {
        "image": "Kmeans_Silhouette_Score_Pattern.png",
        "description": "Seven clusters appear to be a suitable choice."
    },
    "K-Means Clustering": {
        "image": "Kmeans_PCA.png",
        "description": "This section visualizes the clusters formed by k-means and the principal components. The k is chosen to be 7 by considerations in the elbow method, the Silhouette score, and practical reasons."
    },
}

# Define campaign sections for the dashboard
campaign_sections = {
    "Feature Distribution": {
        "image": "num_distribution.png",
        "description": ""
    },
    "Correlation Analysis": {
        "image": "num_corr.png",
        "description": "'Previous' is negative when 'pdays' is negative."
    },
    "Classifier Comparison": {
        "image": "confusion.png",
        "description": "True positive rate is an important indicator for this imbalanced dataset. Examining evaluators including accuracy, precision, recall, ROC_AUC, and so on, the gradient boosting is the best performer among the models."
    },
}


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


# Route: Segmentation Dashboard
@app.route('/segmentation')
def segmentation():
    return render_template('dashboard.html', sections=segmentation_sections)


# Route: Campaign Dashboard
@app.route('/campaign')
def campaign():
    return render_template('campaign.html', sections=campaign_sections)


# Route: Display Images
@app.route('/images/<filename>')
def display_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


# Route: Real-Time Segmentation Prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = os.path.join(MODEL_FOLDER, 'real_time_segmentation.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    if request.method == 'POST':
        data = {
            'housing': [int(request.form['housing'])],
            'loan': [int(request.form['personal'])],
            'cd_account': [int(request.form['cd'])],
            'securities': [int(request.form['securities'])],
            'TransactionFrequency': [float(request.form['transaction_frequency'])],
            'Recency': [float(request.form['Recency'])],
            'AverageTransactionAmount': [float(request.form['AverageAmount'])]
        }

        df = pd.DataFrame(data)
        cluster = model.predict(df)
        return render_template('result.html', cluster=cluster)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)