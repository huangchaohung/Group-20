import json
import random
import pandas as pd
import os

random.seed(3101)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def dynamic_email_modifier(recent_email_result, lr=0.5):
    email_a = pd.read_csv(os.path.join(BASE_DIR, '../data/email_data/email_a.csv'))
    email_b = recent_email_result
    recent_email_result.to_csv(os.path.join(BASE_DIR, '../data/email_data/email_b.csv'), index=False)

    email_a_feature_path = os.path.join(BASE_DIR, '../data/email_data/email_a_features.json')
    email_b_feature_path = os.path.join(BASE_DIR, '../data/email_data/email_b_features.json')

    with open(email_a_feature_path) as f:
        email_a_feature = json.load(f)
    with open(email_b_feature_path) as f:
        email_b_feature = json.load(f)

    mutually_exclusive = eval(open(os.path.join(BASE_DIR, '../data/email_data/mutually_exclusive.txt')).read())

    success_rate_a, success_rate_b = __calculate_email_percentage(email_a, email_b)

    # Determine if adjustments are needed
    if success_rate_a > success_rate_b:
        email_b_feature = __adjust_email_features(
            email_b_feature, email_a_feature, success_rate_b, success_rate_a, lr, mutually_exclusive
        )
    else:
        # If email_b has the best result, offer a random configuration
        email_b_feature = __generate_random_features(mutually_exclusive)

    # Save the adjusted features to a new JSON file for download
    download_path = os.path.join(BASE_DIR, '../data/email_data/adjusted_email_b_features.json')
    with open(download_path, "w") as j:
        json.dump(email_b_feature, j)

    differences, similarities = __compare_emails(email_a_feature, email_b_feature)
    return differences, similarities, download_path


def __calculate_email_percentage(email_a, email_b):
    clickrate_a = email_a['click_rate']
    clickrate_b = email_b['click_rate']
    success_rate_a = sum(clickrate_a) / len(clickrate_a)
    success_rate_b = sum(clickrate_b) / len(clickrate_b)
    return success_rate_a, success_rate_b


def __adjust_email_features(email_low, email_high, success_low, success_high, lr, mutually_exclusive_groups):
    new_email = email_low.copy()
    if success_low < success_high:
        for group in mutually_exclusive_groups:
            for feature in group:
                if new_email[feature] != email_high[feature]:
                    new_email[feature] += lr * (email_high[feature] - new_email[feature])
                    new_email[feature] += random.uniform(-0.05, 0.05)
                    new_email[feature] = max(0, min(1, new_email[feature]))

            max_feature = max(group, key=lambda f: new_email[f])
            for feature in group:
                new_email[feature] = 1 if feature == max_feature else 0
    return new_email


def __generate_random_features(mutually_exclusive_groups):
    random_features = {}
    for group in mutually_exclusive_groups:
        for feature in group:
            random_features[feature] = random.randint(0, 1)
        max_feature = max(group, key=lambda f: random_features[f])
        for feature in group:
            random_features[feature] = 1 if feature == max_feature else 0
    return random_features


def __compare_emails(email_a, email_b):
    differences = {}
    similarities = {}
    for feature in email_a:
        if email_a[feature] != email_b[feature]:
            differences[feature] = (email_a[feature], email_b[feature])
        elif email_a[feature] == 1:
            similarities[feature] = (email_a[feature], email_b[feature])
    return differences, similarities
