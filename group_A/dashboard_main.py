from flask import Flask, render_template, send_from_directory, redirect, url_for, request
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Set the path for images
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "image")

# Define the sections for the dashboard
sections = {
    "Correlation Analysis": {
        "image": "characteristic_corr.png",
        "description": "The features are not correlated at all except for transaction frequency and recency - due to the fact that higher frequency implies a more recent transaction record."
    },
    "Elbow Method": {
        "image": "elbow40.png",
        "description": "The Elbow Method is used to determine the optimal number of clusters for k-means clustering. However, too many customer segments is not realistic in practical banking campaign"
    },
    "Silhouette Analysis": {
        "image": "Kmeans_Silhouette_Score_Pattern.png",
        "description": "Seven clusters appear to be a suitable choice."
    },
    "K-Means Clustering": {
        "image": "Kmeans_PCA.png",
        "description": "This section visualizes the clusters formed by k-means and the principal components. The k is chosen to be 7 by considerations in the elbow method, the Silhouette score, and practical reasons"
    },
    "Clustering Models": {
        "image": "Cluster_Model_Comparison.png",
        "description": "Evaluation on models shows the best performance for K-means clustering."
    },
    "Transaction Frequency": {
        "image": "Transaction_Frequency.png",
        "description": ""
    },
    "Transaction Recency": {
        "image": "Recency.png",
        "description": ""
    },
    "Transaction Amount": {
        "image": "Transaction_Amount.png",
        "description": ""
    },
    "Product Ownership": {
        "image": "Product_ownership.png",
        "description": "Clusters have respective specific characteristics, needs and opportunities."
    },
    "Real-time Segmentation": {
        "image": "real_time_segmentation_confusion.png",
        "description": ""
    },
}

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
        "description": "True positive rate is an important indicator for this imbalanced dataset. Examining evaluators including accuracy, precision, recall, ROC_AUC and so on, the gradient boosting is the best performer among the models"
    },
    "ROC-AUC Curve": {
        "image": "name_ROC.png",
        "description": ""
    },
    "Campaign Top Factors": {
        "image": "campaign_feature_importance.png",
        "description": "Top factors that drive the successfulness of this campaign."
    },
    "Duration Distribution": {
        "image": "duration_against_subscription.png",
        "description": "Higher contact duration is crucial to success of the campaign."
    },
    "Previous Campgin Results": {
        "image": "poutcome_summary_table.png",
        "description": "The successful previous campagin often indicates a higher successful rate for this campaign."
    },
}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]
    return render_template('dashboard.html', sections=sections, images=all_images)

@app.route('/campaign')
def campaign():
    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]
    return render_template('campaign.html', sections=campaign_sections, images=all_images)

@app.route('/images/<filename>')
def display_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

with open(os.path.join(os.path.dirname(__file__), 'models/real_time_segmentation.pkl'), 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve customer info from the form

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

    # If GET request, render the input form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)