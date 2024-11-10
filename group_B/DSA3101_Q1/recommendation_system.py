import pickle
import pandas as pd
from datetime import datetime
import json
import time

# Directory path where models and encoders are stored
model_directory = r"recommendation_models"

# Paths to the saved models and dataset
cd_account_pipeline_path = f"{model_directory}/cd_account_xgb_classifier_0.pkl"
loan_pipeline_path = f"{model_directory}/loan_xgb_classifier_0.pkl"
securities_pipeline_path = f"{model_directory}/securities_xgb_classifier_0.pkl"
term_deposit_pipeline_path = f"{model_directory}/term_deposit_xgb_classifier_0.pkl"
all_features_path = f"{model_directory}/features.json"
dataset_path = r"../../data/Combined_dataset.csv"

# Load the models, feature definitions, dataset, encoders, and scaler
cd_account_pipeline = pickle.load(open(cd_account_pipeline_path, "rb"))
loan_pipeline = pickle.load(open(loan_pipeline_path, "rb"))
securities_pipeline = pickle.load(open(securities_pipeline_path, "rb"))
term_deposit_pipeline = pickle.load(open(term_deposit_pipeline_path, "rb"))
all_features = json.load(open(all_features_path))
data = pd.read_csv(dataset_path)

# Load label encoders for categorical features
encoders = {}
for feature in ["job", "marital", "education", "default", "contact", "poutcome", "month"]:
    with open(f"{model_directory}/{feature}_encoder.pkl", "rb") as f:
        encoders[feature] = pickle.load(f)

# Load the scaler
with open(f"{model_directory}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Descriptions for specific features
feature_descriptions = {
    "campaign": "Number of contacts performed during the current campaign.",
    "pdays": "Number of days since the last contact from a previous campaign (-1 if not contacted).",
    "previous": "Number of contacts performed before the current campaign.",
    "poutcome": "Outcome of the previous marketing campaign (e.g., 'success', 'failure', 'unknown', 'other').",
    "duration": "Duration of marketing campaign engagement.",
    "housing": "Subscribe to housing Loan (0/1)"
}

# Specific options for poutcome
poutcome_options = ["success", "failure", "unknown", "other"]

# Products in the recommendation system
all_products = [
    "CD account",
    "Loan",
    "Securities",
    "Term Deposit"
]

# Map product choice to pipelines and features
product_map = {
    "1": (cd_account_pipeline, all_features["cd_account"]),
    "2": (loan_pipeline, all_features["loan"]),
    "3": (securities_pipeline, all_features["securities"]),
    "4": (term_deposit_pipeline, all_features["term_deposit"]),
}


def recommend_product():
    product_chosen = '\n'.join([f"{_i + 1}) {_product}" for _i, _product in enumerate(all_products)])
    product_choice = input(f"Choose a product:\n{product_chosen}\nEnter the number: ")

    if product_choice not in product_map:
        print("Invalid choice. Please select a number from the list.")
        return

    # Retrieve the selected model and its required features
    model, feature_list = product_map[product_choice]

    # Collect and encode user input for each feature
    features = []
    for _feature in feature_list:
        # Display feature descriptions for specific features
        if _feature in feature_descriptions:
            print(f"{_feature}: {feature_descriptions[_feature]}")

        if _feature == 'month':
            # Automatically set to current month and encode it
            current_month = datetime.now().strftime('%b').lower()
            if 'month' in encoders:
                current_month_encoded = encoders['month'].transform([current_month])[0]
                features.append(current_month_encoded)
            else:
                print(f"Error: 'month' encoder not found.")
                return
            print(f"Setting month to current month: {current_month} (encoded as {current_month_encoded})")

        elif _feature == 'day':
            # Automatically set to today's day as a numeric value
            current_day = datetime.now().day
            features.append(current_day)
            print(f"Setting day to today's day: {current_day}")

        elif _feature == "poutcome":
            # Display specific options for poutcome
            print("Options for poutcome:")
            for idx, option in enumerate(poutcome_options, start=1):
                print(f"{idx}) {option}")
            choice = input("Enter the number corresponding to your choice for poutcome: ")
            try:
                answer = poutcome_options[int(choice) - 1]  # -1 to match index
                encoded_answer = encoders[_feature].transform([answer])[0]
                features.append(encoded_answer)
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")
                return
        elif _feature in encoders:  # Use encoder if available
            # Display options for categorical features from encoders
            unique_values = encoders[_feature].classes_
            print(f"Options for {_feature}:")
            for idx, option in enumerate(unique_values, start=1):
                print(f"{idx}) {option}")
            choice = input(f"Enter the number corresponding to your choice for {_feature}: ")
            try:
                answer = unique_values[int(choice) - 1]  # -1 to match index
                encoded_answer = encoders[_feature].transform([answer])[0]
                features.append(encoded_answer)
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")
                return
        else:
            # Prompt for numerical or unencoded input
            answer = input(f"Enter the value for {_feature}: ")
            try:
                answer = float(answer) if "." in answer else int(answer)
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
            features.append(answer)

    # Create a DataFrame for the input data with only the model's selected features
    input_data = dict(zip(feature_list, features))
    feature_df = pd.DataFrame([input_data])

    # Identify columns expected by the scaler
    expected_columns = scaler.feature_names_in_

    # Add any missing columns with a default value (e.g., 0) for those not in feature_df
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    # Reorder columns to match scaler's expected order
    feature_df = feature_df[expected_columns]

    # Apply scaler transformation to the aligned DataFrame
    feature_df_scaled = pd.DataFrame(scaler.transform(feature_df), columns=expected_columns)

    # Ensure only model-required features are used for prediction
    feature_df_scaled = feature_df_scaled.reindex(columns=feature_list)

    print("======== Generating Results ========")
    # Predict using the scaled feature DataFrame
    prediction = model.predict(feature_df_scaled)
    for _ in range(3):
        # Artificially add time to make it look like its loading
        time.sleep(1)
        print(".")

    if prediction[0] == 0:
        print(f"User SHOULD NOT subscribe to {all_products[int(product_choice)-1]}")
    else:
        print(f"User SHOULD subscribe to {all_products[int(product_choice)-1]}")


# Run the recommendation function
if __name__ == "__main__":
    recommend_product()
