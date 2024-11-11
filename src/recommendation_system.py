import pickle
import pandas as pd
from datetime import datetime
import json
import random

# Define absolute paths for models and data
MODEL_DIR = r"C:\Users\limti\PycharmProjects\DSA3101-Group-20\src\recommendation_models"
FEATURES_JSON_PATH = r"C:\Users\limti\PycharmProjects\DSA3101-Group-20\src\recommendation_models\features.json"

# Load models
cd_account_pipeline = pickle.load(open(f'{MODEL_DIR}/cd_account_xgb_classifier_0.pkl', "rb"))
loan_pipeline = pickle.load(open(f'{MODEL_DIR}/loan_xgb_classifier_0.pkl', "rb"))
securities_pipeline = pickle.load(open(f'{MODEL_DIR}/securities_xgb_classifier_0.pkl', "rb"))
term_deposit_pipeline = pickle.load(open(f'{MODEL_DIR}/term_deposit_xgb_classifier_0.pkl', "rb"))

# Load feature definitions from features.json
with open(FEATURES_JSON_PATH, 'r') as f:
    product_features = json.load(f)

# Load encoders for categorical features
encoders = {}
for feature in ["job", "marital", "education", "default", "contact", "poutcome", "month"]:
    encoders[feature] = pickle.load(open(f"{MODEL_DIR}/{feature}_encoder.pkl", "rb"))

# Load scaler
scaler = pickle.load(open(f'{MODEL_DIR}/scaler.pkl', "rb"))

# Feature descriptions without removed options
feature_descriptions = {
    "age": "Age of the customer",
    "job": "Type of job held by the customer",
    "marital": "Marital status of the customer",
    "education": "Education level of the customer",
    "default": "Whether the customer has defaulted on credit (yes/no)",
    "balance": "Account balance of the customer",
    "housing": "Whether the customer has a housing loan (1/0)",
    "contact": "Type of contact communication (e.g., unknown, telephone)",
    "duration": "Duration of last contact in seconds",
    "campaign": "Number of contacts performed during this campaign",
    "pdays": "Number of days since last contact from a previous campaign",
    "previous": "Number of contacts performed before this campaign",
    "poutcome": "Outcome of the previous marketing campaign"
}

# Map product choices to models
product_map = {
    "1": (cd_account_pipeline, product_features["cd_account"]),
    "2": (loan_pipeline, product_features["loan"]),
    "3": (securities_pipeline, product_features["securities"]),
    "4": (term_deposit_pipeline, product_features["term_deposit"]),
}


# Helper function to generate random values for missing features
def generate_random_value(feature_name):
    if feature_name == "AverageTransactionAmount":
        return random.uniform(100, 1000)  # Example range for transaction amount
    elif feature_name == "Recency":
        return random.randint(0, 365)  # Days since last transaction
    elif feature_name == "TransactionFrequency":
        return random.randint(1, 50)  # Number of transactions
    elif feature_name == "contact":
        return encoders["contact"].transform(["unknown"])[0]  # Default to "unknown"
    elif feature_name == "default":
        return encoders["default"].transform(["no"])[0]  # Default to "no"
    else:
        return 0  # Default for any other missing features


def get_recommendation(product_choice, user_data):
    if product_choice not in product_map:
        return {"error": "Invalid product choice"}, 400

    model, feature_list = product_map[product_choice]
    features = {}

    for feature in feature_list:
        value = user_data.get(feature)

        # Handle month feature specifically if it is None or 'None'
        if feature == 'month' and (value is None or value == 'None'):
            current_month = datetime.now().strftime('%b').lower()
            value = encoders["month"].transform([current_month])[0]
            print(f"Auto-filling month with current month: {current_month} (encoded as {value})")

        # Handle day feature specifically if it is None
        elif feature == 'day' and value is None:
            value = datetime.now().day
            print(f"Auto-filling day with today's day: {value}")

        # Handle categorical features with encoders for other cases
        elif feature in encoders:
            try:
                if isinstance(value, str):
                    value = value.lower()

                if value in encoders[feature].classes_:
                    value = encoders[feature].transform([value])[0]
                else:
                    print(f"Warning: Unseen label '{value}' for feature '{feature}'")
                    value = encoders[feature].transform(["unknown"])[0]
            except Exception as e:
                print(f"Encoding error for {feature}: {e}")
                return {"error": f"Encoding error for {feature}"}, 400

        # Generate a random value if the feature is missing
        elif value is None:
            value = generate_random_value(feature)

        features[feature] = value  # Store the feature value in the dictionary

    # Ensure all required columns are present in features before scaling
    for col in scaler.feature_names_in_:
        if col not in features:
            features[col] = generate_random_value(col)

    # Prepare data for prediction, ensuring correct column order
    input_data = pd.DataFrame([features])
    input_data = input_data[scaler.feature_names_in_]
    feature_df_scaled = pd.DataFrame(scaler.transform(input_data), columns=scaler.feature_names_in_)
    feature_df_scaled = feature_df_scaled.reindex(columns=feature_list)

    # Make the prediction
    try:
        prediction = model.predict(feature_df_scaled)
        recommendation = "subscribe" if prediction[0] == 1 else "do not subscribe"
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": f"Prediction error: {str(e)}"}, 500

    return {
        "product": ["CD Account", "Loan", "Securities", "Term Deposit"][int(product_choice) - 1],
        "recommendation": recommendation
    }

