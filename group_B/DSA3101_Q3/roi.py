import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Feature Engineering Function
def load_and_prepare_data(file_path, avg_revenue_per_conversion=60000):
    df = pd.read_csv(file_path)
    
    # Drop redundant columns
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
    df_marketing['Revenue'] = df_marketing['ConversionRate'] * avg_revenue_per_conversion
    df_marketing['ROI'] = (df_marketing['Revenue'] - df_marketing['AdSpend']) / df_marketing['AdSpend']
    
    # Calculate CLV
    df_marketing['CLV'] = (avg_revenue_per_conversion + df_marketing['LoyaltyPoints']) * df_marketing['PreviousPurchases']
    
    # Label Encoding
    gender_encoder = LabelEncoder()
    campaign_type_encoder = LabelEncoder()
    df_marketing['Gender'] = gender_encoder.fit_transform(df_marketing['Gender'])
    df_marketing['CampaignType'] = campaign_type_encoder.fit_transform(df_marketing['CampaignType'])

    # Drop unnecessary columns
    df_marketing.drop(columns=['CustomerID'], inplace=True)
    
    return df_marketing

# Model Training Function with Hyperparameter Tuning
def train_and_tune_model(X_train, y_train, model_type):
    if model_type == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_type == 'LinearRegression':
        model = LinearRegression()
        return model.fit(X_train, y_train)  # No hyperparameter tuning for Linear Regression

    # Randomized Search with Cross Validation for Hyperparameter Tuning
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model

# Save Model Function
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Model Analysis Function
def analyze_models(df, top_features):
    strategies = df['CampaignChannel'].unique()
    results = {}

    for strategy in strategies:
        print(f"========== {strategy} ==========")
        X = df[df['CampaignChannel'] == strategy][top_features]
        y = df[df['CampaignChannel'] == strategy]['ROI']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Dictionary to store models and results
        models = {
            'RandomForest': train_and_tune_model(X_train, y_train, 'RandomForest'),
            'GradientBoosting': train_and_tune_model(X_train, y_train, 'GradientBoosting'),
            'LinearRegression': train_and_tune_model(X_train, y_train, 'LinearRegression')
        }

        best_model_name = None
        best_r2_score = -np.inf
        model_results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"{model_name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
            model_results[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}

            # Save the best model
            if r2 > best_r2_score:
                best_r2_score = r2
                best_model_name = model_name

            # Save model to a pickle file
            save_model(model, f"Models/{model_name}_model_{strategy}.pkl")
        print(f"Best model for {strategy} is {best_model_name} with R2 score: {best_r2_score}\n")
        print(f"ROI mean: {y.mean()}")
        

# Main function to run the complete pipeline
def main():
    # Load and prepare data
    file_path = 'data/digital_marketing_campaign_dataset.csv'
    df = load_and_prepare_data(file_path)

    # Identify top features for model training
    X = df.drop(columns=['ROI', 'CampaignChannel'])
    y = df['ROI']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    top_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(7).index

    # Analyze models for each strategy
    results = analyze_models(df, top_features)
    
    print(results)

if __name__ == "__main__":
    main()
