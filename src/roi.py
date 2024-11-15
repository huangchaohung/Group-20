import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load and Prepare Data Function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=[
        'AdvertisingTool', 'AdvertisingPlatform', 'PagesPerVisit',
        'EmailOpens', 'WebsiteVisits', 'TimeOnSite',
        'SocialShares', 'ClickThroughRate'
    ], inplace=True)

    # Filter by campaign type
    df_marketing = df[df['CampaignType'].isin(['Conversion', 'Retention'])]
    df_marketing = df_marketing.rename(columns = {'RevenueEarned': 'Revenue'})
    # Calculate revenue and ROI
    df_marketing['ROI'] = (df_marketing['Revenue'] - df_marketing['AdSpend']) / df_marketing['AdSpend']
    # Calculate CLV, assuming a standard period of 1 year
    df_marketing['CLV'] = (df_marketing['Revenue'].mean() + df_marketing['LoyaltyPoints']) * df_marketing['PreviousPurchases']

    # Label Encoding
    gender_encoder = LabelEncoder()
    campaign_type_encoder = LabelEncoder()
    df_marketing['Gender'] = gender_encoder.fit_transform(df_marketing['Gender'])
    df_marketing['CampaignType'] = campaign_type_encoder.fit_transform(df_marketing['CampaignType'])

    # Drop unnecessary columns
    df_marketing.drop(columns=['CustomerID'], inplace=True)
    return df_marketing

# Load Model Function
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

"""
# Function to Get Top Features
def get_top_features(df, overall_model_path):
    X = df.drop(columns=['ROI', 'CampaignChannel'])
    rf = load_model(overall_model_path)
    top_features = pd.Series(rf.feature_importances_, index=X.columns).index
    return top_features
"""

# Model Analysis Function
def analyze_models(df, strategy_input):
    # Filter data by the selected strategy

    df_strategy = df[df['CampaignChannel'] == strategy_input]

    X = df_strategy
    y = df_strategy['ROI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_files = {
        'RandomForest': os.path.join(BASE_DIR, f"roi_models/random_forest_model_{strategy_input}.pkl"),
        'GradientBoosting': os.path.join(BASE_DIR, f"roi_models/gradient_boosting_model_{strategy_input}.pkl"),
        'LinearRegression': os.path.join(BASE_DIR, f"roi_models/linear_regression_model_{strategy_input}.pkl")
    }

    best_model_name = None
    best_model = None
    best_r2 = -float('inf')
    best_train_features = None
    # Load and evaluate each model
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            model = load_model(file_path)
            trained_features = model.feature_names_in_
            y_pred = model.predict(X_test[trained_features])
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # print(f"{model_name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

            # Track the best model by R²
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model = model
                best_train_features = trained_features
        else:
            print(f"Model file for {model_name} in strategy '{strategy_input}' not found.")

    if best_model is not None:
        # Use the best model to predict ROI for the given strategy
        best_model_predictions = best_model.predict(X[best_train_features])
        mean_predicted_roi = np.mean(best_model_predictions)
        return {
            "strategy": strategy_input,
            "best_model": best_model_name,
            "mean_predicted_roi": mean_predicted_roi
        }
    else:
        return {
            "strategy": strategy_input,
            "error": f"No valid model found for strategy '{strategy_input}'."
        }

# Function to Calculate and Plot All ROIs
def calculate_and_plot_all_rois(overall_model_path, dataset_path):
    # Load and prepare the data
    df = load_and_prepare_data(dataset_path)

    # Define available strategies
    strategies = ["Email", "PPC", "Social Media", "Referral", "SEO"]
    roi_values = []
    model_names = []

    # Calculate ROI for each strategy
    for strategy in strategies:
        result = analyze_models(df, strategy)

        if "mean_predicted_roi" in result:
            roi_values.append(result["mean_predicted_roi"])
            model_names.append(result["best_model"])
        else:
            roi_values.append(0)  # If there's an error, default ROI to 0
            model_names.append("N/A")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, roi_values, color='skyblue')
    plt.xlabel("Marketing Strategy")
    plt.ylabel("Mean Predicted ROI")
    plt.title("Mean Predicted ROI by Marketing Strategy")

    # Add the best model names as annotations above each bar
    for i, model_name in enumerate(model_names):
        plt.text(i, roi_values[i] + 0.05, model_name, ha='center')

    # Save plot to a BytesIO object and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url
