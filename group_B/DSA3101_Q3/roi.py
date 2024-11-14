import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Feature Engineering Function
def load_and_prepare_data(file_path, ):
    df = pd.read_csv(file_path)
    
    df.drop(
        columns=[
            'AdvertisingTool', 'AdvertisingPlatform', 'PagesPerVisit', 
            'EmailOpens', 'WebsiteVisits', 'TimeOnSite', 
            'SocialShares', 'ClickThroughRate'
        ],
        inplace=True
    )
    
    # Filter by campaign type
    df_marketing = df[df['CampaignType'].isin(['Conversion', 'Retention'])]

    # Calculate revenue and ROI
    df_marketing['ROI'] = (df_marketing['RevenueEarned'] - df_marketing['AdSpend']) / df_marketing['AdSpend'] 
    # Calculate CLV, assuming a standard period of 1 year
    df_marketing['CLV'] = (df_marketing['RevenueEarned'].mean() + df_marketing['LoyaltyPoints']) * df_marketing['PreviousPurchases']
    
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

# Model Analysis Function
def analyze_models(df, top_features, strategy_input):
    # Filter data by the selected strategy
    
    df_strategy = df[df['CampaignChannel'] == strategy_input]

    X = df_strategy[top_features]
    y = df_strategy['ROI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_files = {
        'RandomForest': f"Models/random_forest_model_{strategy_input}.pkl",
        'GradientBoosting': f"Models/gradient_boosting_model_{strategy_input}.pkl",
        'LinearRegression': f"Models/linear_regression_model_{strategy_input}.pkl"
    }

    best_model_name = None
    best_model = None
    best_r2 = -float('inf')  

    # Load and evaluate each model
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            model = load_model(file_path)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            #print(f"{model_name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

            # Track the best model by R²
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model = model
        else:
            print(f"Model file for {model_name} in strategy '{strategy_input}' not found.")

    if best_model is not None:
        # Use the best model to predict ROI for the given strategy
        best_model_predictions = best_model.predict(X)
        mean_predicted_roi = np.mean(best_model_predictions)
        print(f"Predicted ROI for {strategy_input}: {mean_predicted_roi}\n")
    else:
        print(f"No valid model found for strategy '{strategy_input}'.")

# Main function to run the complete pipeline
def main():
    # Load and prepare data
    file_path = 'Data/banking_marketing_strategies.csv'
    df = load_and_prepare_data(file_path)

    # Identify top features for model testing
    X = df.drop(columns=['ROI', 'CampaignChannel'])
    y = df['ROI']
    rf = load_model("Models/random_forest_model_overall.pkl")  # Load an overall RF model to get top features
    top_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(6).index

    # Take marketing strategy input from the user
    strategy_input = input("Enter the marketing strategy (Email, PPC, Social Media, Referral, SEO): ").strip()

    # Check if the input is valid
    valid_strategies = ['Email', 'PPC', 'Social Media', 'Referral', 'SEO']
    if strategy_input not in valid_strategies:
        print(f"Invalid strategy input. Please choose from: {', '.join(valid_strategies)}")
    else:
        # Analyze models for the given strategy
        analyze_models(df, top_features, strategy_input)

if __name__ == "__main__":
    main()
