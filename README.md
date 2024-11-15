
# Project Overview

This project aims to enhance marketing efficiency for a Portuguese bank by developing an AI-driven system to analyze customer data for personalized marketing campaigns. The main objective is to improve customer engagement and conversion rates for the bank's term deposit products through targeted outreach. Using phone-based campaigns, the system utilizes machine learning to segment customers and optimize marketing efforts, ultimately enabling data-driven decision-making within the bank's marketing strategy.

### Key Goals:
- Increase customer engagement and conversion rates.
- Leverage data-driven insights to improve personalized marketing.
- Enhance marketing efficiency through AI and machine learning.

For more details, please refer to the full [project wiki](/project_wiki.md).

## Instructions for Setting Up the Environment and Running the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/huangchaohung/Group-20.git
   cd Group-20
   ```

2. **Install Python Dependencies**:  
   This project uses Python 3.11+. It's recommended to set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Project**:  
   Once dependencies are installed, you can execute the primary scripts using:
   ```bash
   python main.py
   ```

## Repository Structure

```
Group-20/
    ├── data/
    │   ├── email_data/
    │   │   ├── email_a.csv
    │   │   ├── email_a_features.json
    │   │   ├── email_b.csv
    │   │   ├── email_b_features.json
    │   │   └── mutually_exclusive.txt
    │   ├── Bank_Personal_Loan_Modelling.csv
    │   ├── Churn_Modelling.csv
    │   ├── Combined_dataset.csv
    │   ├── digital_marketing_campaign_dataset.csv
    │   ├── features.json
    │   ├── reviews_for_classification.csv
    │   ├── test.csv
    │   ├── test_Data.csv
    │   ├── train.csv
    │   └── train_data.csv
    ├── group_A/
    │   ├── DSA3101 BQ3/
    │   │   └── predict_customer_churn.py
    │   ├── models/
    │   │   └── real_time_segmentation.pkl
    │   ├── segmentation_analysis/
    │   │   ├── balance.py
    │   │   ├── campaign_data_analysis.py
    │   │   ├── campaign_model_evaluation.py
    │   │   ├── characteristic_visualization.py
    │   │   ├── clustering_modeling.py
    │   │   ├── data_prep.py
    │   │   ├── real_time_evaluation.py
    │   │   ├── segment_corr.py
    │   │   └── segmentation_analysis.py
    │   ├── templates/
    │   │   ├── home.html
    │   │   └── style.css
    │   ├── Bonus qn 2 Real-Time Segmentation.ipynb
    │   ├── Bonus qn 3 Predict Customer Churn.ipynb
    │   ├── Draft Answer.docx
    │   ├── bonus_question1.2.py
    │   ├── bonus_question1.py
    │   ├── dashboard_main.py
    │   ├── draft description.docx
    │   └── segmentation with campaign numerical.ipynb
    ├── group_B/
    │   ├── DSA3101_Q1/
    │   │   ├── recommendation_models/
    │   │   │   ├── cd_account_xgb_classifier_0.pkl
    │   │   │   ├── contact_encoder.pkl
    │   │   │   ├── default_encoder.pkl
    │   │   │   ├── education_encoder.pkl
    │   │   │   ├── features.json
    │   │   │   ├── job_encoder.pkl
    │   │   │   ├── loan_xgb_classifier_0.pkl
    │   │   │   ├── martial_encoder.pkl
    │   │   │   ├── month_encoder.pkl
    │   │   │   ├── poutcome_encoder.pkl
    │   │   │   ├── scaler.pkl
    │   │   │   ├── securities_xgb_classifier_0.pkl
    │   │   │   └── term_deposit_xgb_classifier_0.pkl
    │   │   ├── .DS_Store
    │   │   ├── Recommendation_System_notebook.ipynb
    │   │   └── recommendation_system.py
    │   ├── DSA3101_Q2/
    │   │   ├── sample_email_json/
    │   │   │   ├── email_a_initial.csv
    │   │   │   ├── email_a_initial_features.json
    │   │   │   └── email_b_initial_features.json
    │   │   ├── dynamic_email.py
    │   │   └── email_campaign_adjustment_final.ipynb
    │   ├── DSA3101_Q3/
    │   │   ├── data/
    │   │   │   ├── .DS_Store
    │   │   │   └── digital_marketing_campaign_dataset.csv
    │   │   ├── Models/
    │   │   │   ├── .DS_Store
    │   │   │   ├── gradient_boosting_model_Email.pkl
    │   │   │   ├── gradient_boosting_model_PPC.pkl
    │   │   │   ├── gradient_boosting_model_Referral.pkl
    │   │   │   ├── gradient_boosting_model_SEO.pkl
    │   │   │   ├── gradient_boosting_model_Social Media.pkl
    │   │   │   ├── linear_regression_model_Email.pkl
    │   │   │   ├── linear_regression_model_PPC.pkl
    │   │   │   ├── linear_regression_model_Referral.pkl
    │   │   │   ├── linear_regression_model_SEO.pkl
    │   │   │   ├── linear_regression_model_Social Media.pkl
    │   │   │   ├── random_forest_model_Email.pkl
    │   │   │   ├── random_forest_model_PPC.pkl
    │   │   │   ├── random_forest_model_Referral.pkl
    │   │   │   ├── random_forest_model_SEO.pkl
    │   │   │   ├── random_forest_model_Social Media.pkl
    │   │   │   └── random_forest_model_overall.pkl
    │   │   ├── .DS_Store
    │   │   ├── DSA3101_q3.ipynb
    │   │   └── roi.py
    │   ├── synthetic_data_model/
    │   │   ├── cd_account_random_forest_0.pkl
    │   │   └── securities_random_forest_0.pkl
    │   ├── .DS_Store
    │   ├── Email_Campaign_Adjustment.ipynb
    │   ├── ROI.ipynb
    │   └── product_synthetic_generation.ipynb
    ├── image/
    │   ├── Cluster_Model_Comparison.png
    │   ├── CustomerChurnGradientBoostingModelEvaluation.png
    │   ├── Gradient Boosting_confusion.png
    │   ├── KNN_confusion.png
    │   ├── Kmeans_PCA.png
    │   ├── Kmeans_Silhouette_Score_pattern.png
    │   ├── Logistic Regression_confusion.png
    │   ├── PredictCustomerRentention_ShorterVer.png
    │   ├── Production_ownership.png
    │   ├── Random Forest_confutsion.png
    │   ├── Recency.png
    │   ├── SVM_confusion.png
    │   ├── Transaction_Amount.png
    │   ├── Transaction_Frequency.png
    │   ├── campaign_feature_importance.png
    │   ├── characteristic_corr.png
    │   ├── churn_risk_histogram.png
    │   ├── confusion.png
    │   ├── duration_against_subscription.png
    │   ├── elbow15.png
    │   ├── elbow40.png
    │   ├── feature_importance.png
    │   ├── name_ROC.png
    │   ├── num_corr.png
    │   ├── num_distribution.png
    │   ├── poutcome_summary_table.png
    │   ├── predict_customer_churn_high_risk_customers.png
    │   ├── real_time_segmentation_confusion.png
    │   ├── real_time_segmentation_importance.png
    │   └── shap_summary.png
    ├── src/
    │   ├── recommendation_models/
    │   │   ├── cd_account_xgb_classifier_0.pkl
    │   │   ├── contact_encoder.pkl
    │   │   ├── default_encoder.pkl
    │   │   ├── education_encoder.pkl
    │   │   ├── features.json
    │   │   ├── job_encoder.pkl
    │   │   ├── loan_xgb_classifier_0.pkl
    │   │   ├── martial_encoder.pkl
    │   │   ├── month_encoder.pkl
    │   │   ├── poutcome_encoder.pkl
    │   │   ├── scaler.pkl
    │   │   ├── securities_encoder.pkl
    │   │   ├── securities_xgb_classifier_0.pkl
    │   │   ├── term_deposit_encoder.pkl
    │   │   ├── term_deposit_gradient_boosting_0.pkl
    │   │   └── term_deposit_xgb_classifier_0.pkl
    │   ├── roi_models/
    │   │   └── random_forest_model_overall.pkl
    │   ├── segmentation_models/
    │   │   └── real_time_segmentation.pkl
    │   ├── static/
    │   │   ├── app.js
    │   │   └── style.css
    │   ├── templates/
    │   │   ├── base.html
    │   │   ├── campaign.html
    │   │   ├── dashboard.html
    │   │   ├── dynamic_email.html
    │   │   ├── dynamic_email_result.html
    │   │   ├── home.html
    │   │   ├── predict.html
    │   │   ├── recommendation.html
    │   │   ├── result.html
    │   │   └── roi.html
    │   ├── .DS_Store
    │   ├── __init__.py
    │   ├── dynamic_email.py
    │   ├── main.py
    │   ├── predict_customer_churn.py
    │   ├── recommendation_system.py
    │   └── roi.py
    ├── .DS_Store
    ├── .Rhistory
    ├── Dockerfile
    ├── README.md
    ├── data_dictionary.xlsx
    ├── email_marketing_campaigns_with_demographics.csv
    └── requirements.txt
```

## Data Sources and Data Preparation Steps

The project uses a dataset from Kaggle related to the direct marketing campaigns of a Portuguese bank, which targets customers for term deposit products through phone-based outreach. 
Four complementary datasets from Kaggle were used to generate the synthetic data required for this project.

- **Main Dataset**: [Bank Marketing Dataset](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- ***Complementary Dataset 1***: [Bank Personal Loan](https://www.kaggle.com/datasets/mahnazarjmand/bank-personal-loan/data)
- ***Complementary Dataset 2***: [Bank Customer Segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation)
- ***Complementary Dataset 3***: [Online Banking / Financial Review Dataset](https://www.kaggle.com/datasets/yanmaksi/reviews-data-for-classification-model?resource=download)
- ***Complementary Dataset 4***: [Banking Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/versions/1/data)

## Instructions for Building and Running the Docker Container(s)

1. **Build the Docker Image**:  
   Ensure Docker is installed and running on your machine. Then, in the project root, build the Docker image:
   ```bash
   docker build -t bank-marketing-ai -f Dockerfile .
   ```

2. **Run the Docker Container**:  
   Use the following command to run the container, specifying port `5000` (as per the `EXPOSE` statement in the Dockerfile):
   ```bash
   docker run -p 5000:5000 bank-marketing-ai
   ```

3. **Access the Application**:  
   Once the container is running, you can access the application API at `http://localhost:5000`. Use this URL in a browser or with a tool like Postman to test the endpoints.

## API Documentation

The API serves as the interface to retrieve predictions for customer engagement based on demographic and marketing data. Below is a summary of the primary endpoints:

### Endpoints

### Home Page
**`GET /`**  
- **Description**: Renders the homepage.
- **Returns**: HTML page.

### Recommendation Page
**`GET /recommendation`**  
- **Description**: Renders the product recommendation page.
- **Returns**: HTML page.

### ROI Analysis Page
**`GET /roi_analysis`**  
- **Description**: Generates ROI analysis visualization and displays it.
- **Returns**: HTML page with ROI plot.

### Get Product Recommendations
**`POST /recommend`**  
- **Description**: Provides personalized product recommendations based on user data.
- **Request**:
  - JSON body with fields:
    - `product_choice` (string): Product category for recommendation.
    - `user_data` (object): User data dictionary.
- **Response**: JSON object with recommended products.
- **Example**:
  ```json
  {
    "product_choice": "savings",
    "user_data": {
      "age": 30,
      "income": 70000
    }
  }
  ```

### Get Feature Descriptions
**`GET /feature_descriptions`**  
- **Description**: Returns descriptions of features used in the recommendation model.
- **Response**: JSON object with feature descriptions.

### Get Product Features
**`GET /product_features`**  
- **Description**: Returns details about different product features.
- **Response**: JSON object with product features.

### Dynamic Email Modification
**`GET /dynamic_email`**  
- **Description**: Displays dynamic email modification tool.
- **POST** `/dynamic_email`
  - **Description**: Processes a CSV file for dynamic email comparison.
  - **Request**: File upload of CSV with recent email data.
  - **Response**: Renders HTML page with similarities and differences in email data.

### Download Features
**`GET /download_features`**  
- **Description**: Downloads a processed CSV file.
- **Request**: Query parameter `download_path` specifying file path.
- **Response**: CSV file or 404 error if file not found.

### Segmentation Dashboard
**`GET /segmentation`**  
- **Description**: Displays the segmentation analysis dashboard.
- **Returns**: HTML page with various segmentation analysis sections.

### Campaign Dashboard
**`GET /campaign`**  
- **Description**: Displays the campaign analysis dashboard.
- **Returns**: HTML page with campaign analysis sections.

### Display Images
**`GET /images/<filename>`**  
- **Description**: Serves images by filename, used in dashboards.
- **Request**: `filename` path parameter.
- **Returns**: Requested image or 404 error if file not found.

### Real-Time Segmentation Prediction
**`GET /predict`**  
- **Description**: Displays the prediction form for segmentation.
- **POST /predict**
  - **Description**: Uses user-provided data to predict customer segment.
  - **Request**:
    - Form data:
      - `housing` (int): Indicator for housing loan.
      - `loan` (int): Indicator for personal loan.
      - `cd_account` (int): Indicator for certificate of deposit account.
      - `securities` (int): Indicator for securities account.
      - `TransactionFrequency` (float): Transaction frequency.
      - `Recency` (float): Recency of last transaction.
      - `AverageTransactionAmount` (float): Average transaction amount.
  - **Response**: Renders HTML page with the predicted cluster.

---

## Instructions for Dashboard Generation Using New Dataset

The dashboards shown in the application interface of 'main.py' is generated based on the images in the image folder. If you would like to apply new dataset to generate a different dashboard, there is an image generator dedicated for outputing images that create the dashboard: [segmentation_analysis.py](https://github.com/huangchaohung/Group-20/blob/main/group_A/segmentation_analysis/segmentation_analysis.py). With new images plotted, the user can create new dashboard by running main.py. 

