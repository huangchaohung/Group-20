import json
import random
import pandas as pd
random.seed(3101)
def dynamic_email_modifier(recent_email_result, lr=0.5):
    """
    Create a dynamic email modifier that adjusts based on click rates.
    """
    # Load current email results
    email_a = pd.read_csv('../data/email_data/email_a.csv')
    email_b = recent_email_result
    recent_email_result.to_csv('../data/email_data/email_b.csv', index=False)

    # Load feature configurations for each email
    email_a_feature_path = r'../data/email_data/email_a_features.json'
    email_b_feature_path = r'../data/email_data/email_b_features.json'

    with open(email_a_feature_path) as f:
        email_a_feature = json.load(f)

    with open(email_b_feature_path) as f:
        email_b_feature = json.load(f)

    # Load mutually exclusive groups
    mutually_exclusive = eval(open('../data/email_data/mutually_exclusive.txt', 'r').read())

    # Calculate click rates
    success_rate_a, success_rate_b = __calculate_email_percentage(email_a, email_b)
    print("Email A Click Rate:", success_rate_a)
    print("Email B Click Rate:", success_rate_b)

    # Compare and adjust features
    if success_rate_a > success_rate_b:
        adjusted_email_b_feature = __adjust_email_features(
            email_b_feature, email_a_feature, success_rate_b, success_rate_a, lr, mutually_exclusive
        )
        email_b_feature = adjusted_email_b_feature.copy()
    else:
        adjusted_email_a_feature = __adjust_email_features(
            email_a_feature, email_b_feature, success_rate_a, success_rate_b, lr, mutually_exclusive
        )
        email_a_feature = email_b_feature.copy()
        email_b_feature = adjusted_email_a_feature.copy()
        email_b.to_csv('../data/email_data/email_a.csv', index=False)

    # Save the adjusted features
    with open(email_a_feature_path, "w") as j:
        json.dump(email_a_feature, j)
    with open(email_b_feature_path, "w") as j:
        json.dump(email_b_feature, j)

    # Output differences for diagnostic
    differences, similarities = __compare_emails(email_a_feature, email_b_feature)
    print("====== Changes Made ======")
    for _feature, _changes in differences.items():
        print(f"{_feature}: {_changes[0]} -> {_changes[1]}")

def __calculate_email_percentage(email_a, email_b):
    # Calculate click rate averages
    clickrate_a = email_a['click_rate']
    clickrate_b = email_b['click_rate']

    success_rate_a = sum(clickrate_a) / len(clickrate_a)
    success_rate_b = sum(clickrate_b) / len(clickrate_b)

    return success_rate_a, success_rate_b

def __adjust_email_features(email_low, email_high, success_low, success_high, lr, mutually_exclusive_groups):
    # Create a new dictionary to avoid mutating the input
    new_email = email_low.copy()
    if success_low < success_high:
        for group in mutually_exclusive_groups:
            for feature in group:
                if new_email[feature] != email_high[feature]:
                    # Adjust feature closer to the high-performing email
                    new_email[feature] += lr * (email_high[feature] - new_email[feature])

                    # Add small random noise to avoid exact matching
                    new_email[feature] += random.uniform(-0.05, 0.05)

                    # Ensure binary values stay between 0 and 1
                    new_email[feature] = max(0, min(1, new_email[feature]))

            # Ensure only one feature in each mutually exclusive group is set to 1
            max_feature = max(group, key=lambda f: new_email[f])
            for feature in group:
                new_email[feature] = 1 if feature == max_feature else 0

    return new_email

def __compare_emails(email_a, email_b):
    # Compare features between two emails to detect differences
    differences = {}
    similarities = {}
    for feature in email_a:
        if email_a[feature] != email_b[feature]:
            differences[feature] = (email_a[feature], email_b[feature])
        elif email_a[feature] == 1:
            similarities[feature] = (email_a[feature], email_b[feature])
    return differences, similarities
