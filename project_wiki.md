# Home Page

## Project Overview and Objectives
Banks face challenges in targeting marketing effectively due to limited personalization, resulting in low engagement and inefficient use of resources. This project seeks to develop an AI-driven system using machine learning to personalize marketing campaigns, aiming to improve engagement, conversion rates, and support data-driven marketing strategies.

## Team Members
- **Data Scientists (Group A)**: Develop machine learning models and perform data analysis.
- **Data Scientists (Group B)**: Develop machine learning models and perform data analysis.

## Quick Links to Key Sections
- [Business Understanding](#business-understanding)
- [Company Analysis](#company-analysis)
- [Industry Analysis](#industry-analysis)
- [Competitive Landscape](#competitive-landscape)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Technical Implementation](#technical-implementation)
- [Analytical Findings](#analytical-findings)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [Lessons Learned](#lessons-learned)
- [References](#references)
- [Appendices](#appendices)

---

# Business Understanding

## Detailed Description of the Business Problem
The project addresses the lack of personalization in bank marketing campaigns, which leads to low engagement and inefficient resource use. Traditional methods fail to leverage data effectively to meet individual customer needs.

## Key Stakeholders and Their Needs
- **Marketing Team**: Requires tools for creating effective, personalized campaigns.
- **Bank Executives**: Seek better marketing ROI and enhanced customer satisfaction.
- **IT Team**: Needs secure, reliable data systems.
- **Customers**: Prefer relevant marketing that meets their interests and needs.

## Success Criteria for the Project
- **Increased Engagement**: Higher engagement in personalized marketing campaigns.
- **Higher Conversion Rates**: More conversions from targeted campaigns.
- **Data-Driven Decisions**: Marketing strategies informed by model insights.
- **Resource Efficiency**: Cost-effective campaigns due to targeted personalization.

---

# Company Analysis

## History and Background
The Portuguese bank involved in this project is a leading institution in retail banking, known for prioritizing customer-centric service across Portugal. Its traditional marketing strategy includes using direct marketing campaigns to promote term deposits through personalized customer interactions (Rathi, 2020).

## Mission, Vision, and Core Values
The bank's mission centers around providing secure and reliable banking solutions while fostering financial empowerment. Its values emphasize transparency, integrity, and a commitment to excellent customer service, which helps build long-term customer relationships and financial well-being (Rathi, 2020).

## SWOT Analysis
- **Strengths**: Strong brand recognition, extensive customer data, and a skilled marketing team.
- **Weaknesses**: Limited effectiveness of some marketing efforts, heavy reliance on phone-based marketing, and privacy concerns.
- **Opportunities**: Expanding digital channels, leveraging machine learning for insights, and reaching new demographics.
- **Threats**: Regulatory restrictions on data use, rising competition from digital-first banks, and market saturation (Tariq, 2022).

## Strategy Alignment with Goals
The bank's strategy of direct phone-based marketing aims to increase term deposits by facilitating personal, targeted communication. Integrating AI-driven customer segmentation aligns with its goal of enhancing engagement and campaign effectiveness (Silva & Santos, 2021).

## Market Position and Competitive Advantage
As a leading bank in Portugal, this institution leverages its extensive customer base and local insights. This established reputation offers a competitive advantage over newer, digital-only entrants (Rathi, 2020; Tariq, 2022).

---

# Industry Analysis

## Key Trends and Developments
The Portuguese banking sector is rapidly digitalizing, with a push toward AI-driven solutions to enhance customer engagement and marketing efficiency. Phone-based campaigns remain common but face limitations in reach and cost-effectiveness, making data-driven methods essential (Silva & Santos, 2021; Tariq, 2022).

## Main Competitors
The bank’s main competitors include both national and international banks with digital-first offerings that appeal to tech-savvy consumers, presenting a competitive challenge (Tariq, 2022).

## SWOT Analysis
- **Strengths**: Strong customer trust, regulatory experience, and comprehensive service offerings.
- **Weaknesses**: Limited digital outreach and slow adaptation to modern technology.
- **Opportunities**: Integrating AI into marketing, partnering with tech companies, and exploring new customer demographics.
- **Threats**: Growing fintech competition, stringent data regulations, and economic fluctuations (Silva & Santos, 2021).

## Regulatory and Legal Considerations
The bank must comply with GDPR and other regulations governing data usage and customer privacy in Portugal. These laws heavily influence the bank’s direct marketing practices, making transparency a key factor (Tariq, 2022).

## Economic Factors
Portugal's economic stability affects customer demand, with low-interest rates pushing banks to offer attractive deposit options. Economic drivers also shape the bank's focus on secure, long-term deposits to attract cautious investors (Rathi, 2020; Tariq, 2022).

---

# Competitive Landscape

## Direct and Indirect Competitors
Direct competitors include other Portuguese banks offering similar deposit options, while indirect competition comes from fintech companies with advanced digital offerings (Tariq, 2022).

## Competitor Strengths and Weaknesses
Competitors often excel in digital outreach but may lack the same customer loyalty and local knowledge, providing the bank with a unique advantage (Silva & Santos, 2021).

## Product Comparison
The bank’s deposit products are competitive in rates and terms but could be further differentiated through more personalized service and enhanced marketing outreach (Rathi, 2020).

## Competitor Strategies and Differentiation
Many competitors emphasize digital and AI-driven campaigns, which the bank aims to emulate to modernize its traditional phone-based strategy while retaining its focus on customer trust (Tariq, 2022).

---

# Data Understanding

## Data Sources and Collection Methods
- **Banking Dataset - Marketing Targets**: The main dataset containing bank customers' demographics and the bank products they consume. https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets
- **Bank Personal Loan**: Complementary dataset for synthetic additional columns for other bank products. https://www.kaggle.com/datasets/mahnazarjmand/bank-personal-loan/data
- **Bank Customer Segmentation** Complementary dataset for synthetic additional columns for other customer engagement data. https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation
- **Online Banking / Financial Review Dataset** Complementary dataset for customer comments analysis. https://www.kaggle.com/datasets/yanmaksi/reviews-data-for-classification-model?resource=download

## Initial Data Exploration Findings
- **Demographics**: Younger customers are more responsive to digital campaigns.
- **Transaction Patterns**: Specific transactions correlate with product preferences.
- **Engagement Trends**: Higher engagement correlates with higher conversion rates.

## Data Quality Assessment
- **Missing Data**: Some gaps in demographic and transaction data.
- **Data Consistency**: Formatting inconsistencies in engagement metrics.
- **Outliers**: Transaction and engagement outliers identified for review.

---

# Data Preparation

## Data Cleaning Processes
- Removal of duplicates, handling of missing values, and standardizing data formats.


### Synthetic Data Generation

To enhance our dataset, we generated synthetic data for two additional columns, supporting [goal, e.g., enhanced product prediction and segmentation]. This process involved training a machine learning model with overlapping columns from two datasets to predict and generate the missing product data in our current dataset.

#### Methodology Summary

1. **Identifying Overlapping Columns**:
   - Overlapping columns present in both the source dataset (which contained the target product data) and the current dataset were used as features for training a machine learning model to predict the missing product columns.

2. **Model Training and Prediction**:
   - Using [mention model type, e.g., Random Forest Regressor], we trained a model to predict the product columns based on overlapping features, thereby generating synthetic data aligned with real data distributions.

3. **Validation and Quality Check**:
   - Validation was conducted to ensure that the synthetic columns aligned well with existing data and maintained consistent distributions, reducing potential bias.

For full details, refer to **Appendix C: Synthetic Data Generation Notebook**.


## Feature Engineering

In this project, feature engineering was performed to enhance the dataset and prepare it for analysis and modeling. Key steps included renaming columns, transforming categorical variables, and creating new features. Below is a summary of the main feature engineering steps applied:

1. **Column Renaming**:
   - The target variable `y` was renamed to `term_deposit` to provide clarity, making it easier to interpret and reference in subsequent analysis.

2. **Categorical Encoding**:
   - Categorical columns such as `job`, `marital`, `education`, `month`, and others were identified and prepared for analysis.
   - Count plots were used for categorical variables to visualize the distribution, ensuring a better understanding of class imbalances or dominant categories.

3. **Product Segmentation**:
   - The dataset includes various products: `loan`, `term_deposit`, `cd_account`, and `securities`. These products were isolated to analyze individual customer engagement patterns and predict preferences for specific products.

4. **Feature Engineering for Temporal Variables**:
   - The `month` column was ordered and visualized to understand seasonality and monthly trends, which can help improve model performance by capturing temporal patterns.

5. **Numerical Feature Selection**:
   - Numerical columns were extracted for separate analysis and visualization, providing insights into features like `balance`, `day`, `duration`, `campaign`, and others. These variables were checked for outliers, skewness, and distributions.

### Example Code Snippet
In the notebook, the following code was used to process and visualize categorical and numerical columns:

```python
# Categorical feature identification
categorical_columns = list(data.select_dtypes(include=['object']).columns) + ["term_deposit", "cd_account", "securities"]

# Numerical feature extraction
numeric_columns = [_col for _col in data.columns if _col not in categorical_columns]

# Renaming target variable
data.rename({"y": "term_deposit"}, axis="columns", inplace=True)
```

## SQL Queries
- Examples of queries used for aggregating transaction data, joining tables, and filtering by recent activity.

## Final Dataset Structure
A cleaned, structured dataset with essential features for model training and analysis.

---

# Modeling

## Modeling Techniques Considered
- Logistic regression, decision trees, random forests, and gradient boosting.

## Model Selection Criteria
- Models were chosen based on interpretability, accuracy, computational efficiency, and their suitability for handling categorical and numerical data.

## Detailed Description of the Modeling Process

1. **Feature Selection**:
   - **Recursive Feature Elimination (RFE)** was used to identify the most relevant features for each product separately. This was necessary as some features had high dimensionality and could introduce noise into the model if not carefully selected.

2. **Data Preprocessing**:
   - **Categorical Encoding**: Label encoding was applied to categorical columns (`job`, `marital`, `education`, etc.), saving the encoders for consistency across model training and future predictions.
   - **Scaling**: Standard scaling was used for numerical features to normalize the data. This step helps ensure model stability and improve performance, particularly for models sensitive to feature scales.

3. **Correlation Analysis**:
   - **Correlation Check**: A correlation matrix was generated to assess relationships between product columns (`loan`, `term_deposit`, `cd_account`, and `securities`). Results showed low correlation, confirming the need to approach feature selection individually for each product model.

## Detailed Description of the Chosen Model(s)
- **Gradient Boosting Model**: Selected for its high accuracy in predicting conversion likelihood and ability to handle complex feature interactions effectively.
- **Random Forest**: Applied as an additional model, valued for interpretability and robustness against overfitting, especially when used with RFE-selected features.

## Model Performance Metrics and Interpretation
- **Precision**: Measures the model’s ability to correctly identify true positives, critical for evaluating marketing effectiveness.
- **Recall**: Indicates the model’s success in capturing all positives, which is essential for understanding campaign reach.
- **F1 Score**: Provides a balanced metric, combining precision and recall for comprehensive evaluation.
- **R² Score** (Random Forest and Gradient Boosting Models): Used to evaluate the overall fit of the model, especially for regression tasks related to campaign effectiveness.

## Example Code Snippets

### Feature Selection and Preprocessing

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Encode categorical variables
for col in categorical_columns:
    encoder = LabelEncoder()
    processed_data[col] = encoder.fit_transform(processed_data[col])
    with open(f"{model_directory}/{col}_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

# Scale numerical columns
scaler = StandardScaler()
processed_data[numerical_columns] = scaler.fit_transform(processed_data[numerical_columns])

# Feature selection with RFE for the 'term_deposit' model
model = GradientBoostingClassifier()
rfe = RFE(model, n_features_to_select=10)
X_rfe = rfe.fit_transform(processed_data, y['term_deposit'])
```

---

# Evaluation

## Evaluation of Model Performance Against Business Objectives
The models were evaluated to ensure they meet the business objectives of increasing customer engagement and conversion rates. Key performance metrics used for evaluation include:

- **Accuracy**: Provides an overall measure of correct predictions.
- **F1 Score**: Used as a balanced metric to evaluate both precision and recall, especially important given the potential class imbalance.
- **Recall**: Measures the ability of the model to capture all true positive cases, ensuring maximum reach in marketing campaigns.
- **Precision**: Indicates the proportion of correct positive predictions, which is essential to reduce false positives in targeted recommendations.

These metrics demonstrated that the chosen models, particularly those using SMOTE for class balance, effectively aligned with business objectives, showing improvements in engagement and conversion predictions.

## Model Comparison and Insights
A **Grid Search with Cross-Validation** was employed to optimize model parameters, and the **Standard Scaling** and **Pipeline** setup ensured consistent preprocessing across model training and testing.

- **Gradient Boosting Model**: Exhibited high F1 and accuracy scores, making it suitable for balanced performance.
- **Random Forest**: Provided strong precision and recall scores, with feature interpretability aiding in customer segmentation insights.

## Limitations of the Current Approach
- **Class Imbalance**: While SMOTE helps balance classes, it may introduce synthetic data noise, potentially impacting model robustness.
- **Dependence on Static Features**: The models might underperform if customer behaviors change frequently, as they rely on historical data without time-series analysis.

## Suggestions for Model Improvements
- **Implement Real-Time Model Updates**: Dynamic updates based on new customer data could improve accuracy.
- **Incorporate Additional Data**: Time-based features could be added for a temporal perspective on customer behaviors.

## Code Snippet for Model Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE

def model_pipeline(X, y, model, param_grid=None):
    smote = SMOTE(random_state=3101)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Define pipeline and grid search for parameter tuning
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_resampled, y_resampled)
        best_pipeline = grid_search.best_estimator_
    else:
        best_pipeline = pipeline.fit(X_resampled, y_resampled)

    # Metrics on test data
    y_pred = best_pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1 Score": f1_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
    }
    return metrics
```
### Further Analysis and Visuals
For a more detailed evaluation, please refer to the [Model Evaluation Section in the Notebook]([link/to/notebook](https://github.com/huangchaohung/Group-20/edit/main/project_wiki.md)).

---

# Deployment

## API Documentation

### Endpoints
- `/predict`: Accepts customer data for conversion likelihood prediction.

### Request/Response Formats
- **Request**: JSON with customer attributes.
- **Response**: JSON with predicted conversion probability.

### Usage Examples
- Sample code for calling the API and interpreting responses.

## Instructions for Running the Docker Container
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

---

# Technical Implementation

## Repository Structure
```
Group-20/
    ├── data/
    │   ├── email_data/
    │   │   ├── email_a.csv
    │   │   ├── email_a_features.json
    │   │   ├── email_b.csv
    │   │   └── mutually_exclusive.txt
    │   ├── Bank_Personal_Loan_Modelling.csv
    │   ├── Combined_dataset.csv
    │   ├── digital_marketing_campaign_dataset.csv
    │   ├── test.csv
    │   ├── test_Data.csv
    │   ├── train.csv
    │   └── train_data.csv
    ├── group_A/
    │   ├── Bonus qn 2 Real-Time Segmentation.ipynb
    │   ├── Bonus qn 3 Predict Customer Churn.ipynb
    │   ├── Draft Answer.docx
    │   ├── draft description.docx
    │   └── segmentation with campaign numerical.ipynb
    ├── group_B/
    │   ├── DSA3101_Q1/
    │   │   ├── recommendation_models/
    │   │   │   ├── cd_account_xgb_classifier_0.pkl
    │   │   │   ├── contact_encoder.pkl
    │   │   │   ├── default_encoder.pkl
    │   │   │   ├── features.json
    │   │   │   ├── job_encoder.pkl
    │   │   │   ├── loan_xgb_classifier_0.pkl
    │   │   │   ├── martial_encoder.pkl
    │   │   │   ├── month_encoder.pkl
    │   │   │   ├── poutcome_encoder.pkl
    │   │   │   ├── scaler.pkl
    │   │   │   ├── securities_xgb_classifier_0.pkl
    │   │   │   └── term_deposit_xgb_classifier_0.pkl
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
    │   │   │   └── digital_marketing_campaign_dataset.csv
    │   │   ├── .DS_Store
    │   │   ├── DSA3101_q3.ipynb
    │   │   └── model.ipynb
    │   ├── synthetic_data_model/
    │   │   ├── cd_account_random_forest_0.pkl
    │   │   └── securities_random_forest_0.pkl
    │   ├── .DS_Store
    │   ├── Email_Campaign_Adjustment.ipynb
    │   ├── ROI.ipynb
    │   └── product_synthetic_generation.ipynb
    ├── src/
    │   ├── __init__.py
    ├── .DS_Store
    ├── Dockerfile
    ├── README.md
    ├── data_dictionary.xlsx
    ├── email_marketing_campaigns_with_demographics.csv
    ├── main.py
    ├── project_wiki.md
    └── requirements.txt
```

## Setup Instructions
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

## Dependency Management
﻿Flask==3.0.3
Werkzeug==3.1.3
pandas==2.2.2
numpy==2.1.0
scikit-learn==1.5.1
matplotlib==3.9.2
xgboost==2.1.2

## Code Style Guide Adherence
- Follows PEP8 and internal coding standards.

---

# Analytical Findings

## Subgroup A
- Question 1:
- Question 2:
- Question 3:
- Optional 1:
- Optional 2:
- Optional 3:

## Subgroup B
- Question 1:
- Question 2: 
What strategies can we implement to optimize our marketing campaigns in real-time? Create an algorithm for dynamic campaign adjustment based on real-time performance metrics. Simulate the impact of proposed adjustments on campaign effectiveness.
- Question 3: 3. How can we measure and maximize the ROI of our personalized marketing efforts? Develop a model to calculate and predict ROI for different marketing strategies. Incorporate factors such as customer lifetime value, campaign costs, and conversion rates.
- Optional 1:
- Optional 2:
- Optional 3:

---

# Recommendations

## Prioritized List of Recommendations
1. Focus on high-engagement customers for targeted campaigns.
2. Expand digital marketing to younger demographics.

## Implementation Roadmap
- Step-by-step guide for rolling out personalized campaigns based on findings.

## Expected Impact of Each Recommendation
- Anticipated improvements in engagement and conversion rates for each recommendation.

---

# Marketing Campaigns (Subgroup B: Question 2)

The marketing campaign chosen to reach out to potential clients would be through the use of emails. First, through our product recommendation system developed in the first question, we split our customers into the demographics based on what products we should recommend to them. For example, assuming we have identified a group of customers who should take up CD accounts, we will send an email to this group of individuals promoting the benefits of CD accounts, encouraging them to sign up. 

## Email features
An email will consist of several features. Heres how an email will be defined in our campaign.

```python
{
    "Number of Images_0": 1,
    "Number of Images_1-2": 0,
    "Number of Images_3-5": 0,
    "Number of Images_6+": 0,

    "Text Length_Short": 1,
    "Text Length_Medium": 0,
    "Text Length_Long": 0,
    "Text Length_Very_Long": 0,

    "Font Emphasis_None": 1,
    "Font Emphasis_Bold": 0,
    "Font Emphasis_Italic": 0,
    "Font Emphasis_Both": 0,

    "Wording Focus_High Returns": 1,
    "Wording Focus_Stable Income": 0,
    "Wording Focus_How Money is Used": 0,
    "Wording Focus_Security": 0,

    "Tone_Formal": 1,
    "Tone_Conversational": 0,
    "Tone_Urgent": 0,
    "Tone_Friendly": 0,

    "Subject Line Type_Promotional": 1,
    "Subject Line Type_Informative": 0,
    "Subject Line Type_Personalized": 0,
    "Subject Line Type_Question-based": 0,

    "Image Type_Professional": 1,
    "Image Type_Lifestyle": 0,
    "Image Type_Infographic": 0,
    "Image Type_Icons": 0,

    "Layout Complexity_Simple": 1,
    "Layout Complexity_Moderate": 0,
    "Layout Complexity_Complex": 0,

    "Target Audience_High Salary": 1,
    "Target Audience_Low Salary": 0,
    "Target Audience_Mid Salary": 0,
    "Target Audience_Young Professionals": 0,

    "Font Size_Small": 1,
    "Font Size_Medium": 0,
    "Font Size_Large": 0,

    "Color Contrast_High": 1,
    "Color Contrast_Medium": 0,
    "Color Contrast_Low": 0,

    "Testimonial Inclusion_None": 1,
    "Testimonial Inclusion_Short": 0,
    "Testimonial Inclusion_Detailed": 0,
    "Testimonial Inclusion_User Reviews": 0,

    "Data-Driven Content_None": 1,
    "Data-Driven Content_Graphs": 0,
    "Data-Driven Content_Projections": 0,
    "Data-Driven Content_Comparisons": 0,

    "Offer Type_Fixed Rate": 1,
    "Offer Type_Variable Rate": 0,
    "Offer Type_Bonus Rate": 0,
    "Offer Type_Tiered Rate": 0,

    "Email Length_Short": 1,
    "Email Length_Medium": 0,
    "Email Length_Long": 0,

    "Personalization_None": 1,
    "Personalization_Subject": 0,
    "Personalization_Body and Subject": 0,
    "Personalization_Offer": 0,

    "Urgency Tone_None": 1,
    "Urgency Tone_Mild Urgency": 0,
    "Urgency Tone_Strong Urgency": 0,
    "Urgency Tone_Exclusive Offer": 0,

    "Bullet Points_None": 1,
    "Bullet Points_Few": 0,
    "Bullet Points_Many": 0
}
```

## Example of an email

An example of how this would look is like so 

```
Subject Line: Secure High Returns with Our Fixed-Rate CD Accounts

Body:

[Insert Professional Image, e.g., a secure vault or bank logo]

Dear Valued Customer,

Are you looking for a secure and rewarding way to grow your wealth? Our **Certificate of Deposit (CD) accounts** offer a reliable investment opportunity with **fixed interest rates** designed to help you achieve your financial goals.

With our CD accounts, you can:

- **Enjoy guaranteed high returns** over the term of your deposit.
- Benefit from a **safe, low-risk investment option**.

Whether you’re planning for future expenses or simply looking for a dependable growth solution, our CD accounts offer stability and a secure return on your investment. Open an account today and let your money work for you.

Warm regards,
[Your Bank Name]
```

## Email improvement

From there, we will analyze the results of this email through different metrics such as "Click rates", "Subscription rates". In the spirit of improvement, we will create another email that we will send, and if this new email performs worse, heres where our adjustment comes in to create better emails. 

Firstly, we identify a few groups of features that we deem are extremely important and have higher impact. The groups are as shown below.

```python
high_impact_groups = [
    ['Tone_Formal', 'Tone_Conversational', 'Tone_Urgent', 'Tone_Friendly'],
    ['Wording Focus_High Returns', 'Wording Focus_Stable Income', 'Wording Focus_How Money is Used', 'Wording Focus_Security'],
    ['CTA Position_Early', 'CTA Position_Middle', 'CTA Position_Late']
]
```

Next, we modify the email that has poorer performance to resemble the better performing email in terms of features. 

```python
def adjust_email_features(email_low, email_high, success_low, success_high, mutually_exclusive_groups, high_impact_groups):
    new_email = email_low.copy()
    
    if success_low < success_high:
        # Calculate feature similarity: count how many features are the same between the emails
        matching_features = sum(1 for feature in email_low if email_low[feature] == email_high[feature])
        total_features = len(email_low)
        similarity_ratio = matching_features / total_features  # Ratio of matching features (0 to 1)

        for group in mutually_exclusive_groups:
            # Determine the learning rate based on the group impact
            lr = 0.7 if group in high_impact_groups else 0.3

            for feature in group:
                if new_email[feature] != email_high[feature]:
                    # Adjust the feature closer to the high-performing email
                    new_email[feature] += lr * (email_high[feature] - new_email[feature])

                    # Set noise based on feature similarity: more similarity means larger noise
                    baseline_noise = 0.1
                    noise_scaling = baseline_noise + 0.1 * similarity_ratio  # Higher similarity increases noise
                    random_noise = random.uniform(-noise_scaling, noise_scaling)
                    new_email[feature] += random_noise

                    # Ensure binary values stay between 0 and 1
                    new_email[feature] = max(0, min(1, new_email[feature]))

            # Ensure mutual exclusivity in the group
            max_feature = max(group, key=lambda f: new_email[f])
            for feature in group:
                new_email[feature] = 1 if feature == max_feature else 0

    return new_email
```

Here the learning rate (lr) is the variable that controls how closely the new email is going to resemble the better email. A higher lr would result in the new email feature being closer in terms of value.

A random value (small noise) has been added to help make sure that the emails will not be the same, this noise scales to become larger depending on how similar the 2 emails are in terms of features to prevent convergence.

From here, the email will be sent out and tested again to evaluate the performance of the email. Asynchronously, we will be collecting data and analyzing to see if the weights for the learning rate need to be adjusted.

## Simulation 

### Email B features

```python
{
    "Number of Images_0": 1,
    "Number of Images_1-2": 0,
    "Number of Images_3-5": 0,
    "Number of Images_6+": 0,

    "Text Length_Short": 1,
    "Text Length_Medium": 0,
    "Text Length_Long": 0,
    "Text Length_Very_Long": 0,

    "Font Emphasis_None": 0,
    "Font Emphasis_Bold": 1,
    "Font Emphasis_Italic": 0,
    "Font Emphasis_Both": 0,

    "Wording Focus_High Returns": 1,
    "Wording Focus_Stable Income": 0,
    "Wording Focus_How Money is Used": 0,
    "Wording Focus_Security": 0,

    "Tone_Formal": 0,
    "Tone_Conversational": 1,
    "Tone_Urgent": 0,
    "Tone_Friendly": 0,

    "Subject Line Type_Promotional": 0,
    "Subject Line Type_Informative": 0,
    "Subject Line Type_Personalized": 1,
    "Subject Line Type_Question-based": 0,

    "Image Type_Professional": 1,
    "Image Type_Lifestyle": 0,
    "Image Type_Infographic": 0,
    "Image Type_Icons": 0,

    "Layout Complexity_Simple": 1,
    "Layout Complexity_Moderate": 0,
    "Layout Complexity_Complex": 0,

    "Target Audience_High Salary": 1,
    "Target Audience_Low Salary": 0,
    "Target Audience_Mid Salary": 0,
    "Target Audience_Young Professionals": 0,

    "Font Size_Small": 0,
    "Font Size_Medium": 0,
    "Font Size_Large": 1,

    "Color Contrast_High": 1,
    "Color Contrast_Medium": 0,
    "Color Contrast_Low": 0,

    "Testimonial Inclusion_None": 1,
    "Testimonial Inclusion_Short": 0,
    "Testimonial Inclusion_Detailed": 0,
    "Testimonial Inclusion_User Reviews": 0,

    "Data-Driven Content_None": 1,
    "Data-Driven Content_Graphs": 0,
    "Data-Driven Content_Projections": 0,
    "Data-Driven Content_Comparisons": 0,

    "Offer Type_Fixed Rate": 1,
    "Offer Type_Variable Rate": 0,
    "Offer Type_Bonus Rate": 0,
    "Offer Type_Tiered Rate": 0,

    "Email Length_Short": 1,
    "Email Length_Medium": 0,
    "Email Length_Long": 0,

    "Personalization_None": 0,
    "Personalization_Subject": 0,
    "Personalization_Body and Subject": 1,
    "Personalization_Offer": 0,

    "Urgency Tone_None": 0,
    "Urgency Tone_Mild Urgency": 0,
    "Urgency Tone_Strong Urgency": 1,
    "Urgency Tone_Exclusive Offer": 0,

    "Bullet Points_None": 0,
    "Bullet Points_Few": 1,
    "Bullet Points_Many": 0
}
```

This new email sent out had poor performance of about a 0.2 click rate, while the previous email had a click rate of 0.45. As such, we adjust the email to become more like the first email. The results of running this email would be something like this. 

```
Email A Click Rate: 0.432

Email B Click Rate: 0.252

====== Changes Made ======

Font Emphasis_None: 1 -> 0

Font Emphasis_Bold: 0 -> 1

Tone_Formal: 1 -> 0

Tone_Conversational: 0 -> 1

Subject Line Type_Promotional: 1 -> 0

Subject Line Type_Personalized: 0 -> 1

Target Audience_High Salary: 1 -> 0

Target Audience_Mid Salary: 0 -> 1

Font Size_Small: 1 -> 0

Font Size_Large: 0 -> 1

Testimonial Inclusion_None: 1 -> 0

Testimonial Inclusion_Detailed: 0 -> 1

Offer Type_Fixed Rate: 1 -> 0

Offer Type_Variable Rate: 0 -> 1

Personalization_None: 1 -> 0

Personalization_Body and Subject: 0 -> 1

Urgency Tone_None: 1 -> 0

Urgency Tone_Strong Urgency: 0 -> 1

Bullet Points_None: 1 -> 0

Bullet Points_Few: 0 -> 1
```

We can see the changes made and then based on these new features, we can get a new email. 

### First email (poor performer)


```
Subject Line: Earn Secure, High Returns with Our Fixed-Rate CD

Body:

[Professional Image: e.g., secure bank vault or logo]

Hello,

Are you looking for a secure and straightforward way to grow your savings? Our **Fixed-Rate Certificate of Deposit (CD)** offers a reliable investment option with **high, guaranteed returns**. This is a no-fuss, low-risk way to ensure your money works for you.

Why choose our Fixed-Rate CD?

- **Consistent, high returns** with a fixed interest rate
- Stability and security for long-term financial growth
Act now and start building a stable financial future with our trusted Fixed-Rate CD.

Sincerely,
[Your Bank Name]
```


### New improved email


```
Subject Line: [Your Name], Earn Higher Returns with Our Variable Rate CD!

Body:

[Professional Image: e.g., a business professional reviewing investment options]

Hello [First Name],

Are you ready to make the most of your savings? With our **Variable Rate Certificate of Deposit (CD)**, you can enjoy a unique opportunity to earn **higher returns as market rates rise**. It’s a flexible and rewarding way to grow your wealth over time.

Why choose our Variable Rate CD?

- **High earning potential** with market-driven returns
- Flexibility to maximize gains over the deposit term
- Backed by our industry-leading expertise

Here’s what one of our valued customers had to say:

**_"Opening a Variable Rate CD was the best decision I made for my future. The returns have consistently outpaced my previous investments, and I love knowing my money is in safe hands with [Bank Name]."_**

This offer won’t last long—take advantage of this exclusive opportunity to make your money work harder for you.

Best regards,
[Your Bank Name]
```

# ROI Prediction For Marketing Campaigns (Subgroup B: Question 3)

## Project Overview
This project aims to measure and maximize the ROI of personalized marketing efforts for a banking institution. Our campaign involves multiple digital strategies to engage customers and drive conversions while focusing on long-term retention. By calculating ROI and analyzing feature importance, we aim to determine which strategies most effectively enhance revenue and customer value.

## Marketing Strategy
We utilized five primary marketing channels:

1. **Email** : Email ads tailored to specific customer segments.
2. **PPC**: Pay per clicks ads tailored to specific customer segments.
3. **Social Media**: Building brand awareness and fostering customer engagement.
4. **Referral**: Acquiring customers through existing clients.
5. **SEO**: Improving search visibility and credibility.

Each campaign type aligned with different stages of the customer journey:
- **Awareness**: Increasing brand visibility.
- **Consideration**: Educating potential customers.
- **Conversion**: Encouraging sign-ups and account openings.
- **Retention**: Engaging existing customers to foster loyalty.

Our analysis primarily focuses on **Conversion** and **Retention** stages, as they are most relevant to immediate ROI impact and long-term customer value.

## Feature Engineering

We calculated several metrics critical to evaluating marketing effectiveness and customer value.

### 1. ROI Calculation
   - Average Revenue per Conversion: SGD $60,000.
   - Formula:
     ```python
     df_marketing['Revenue'] = df_marketing['ConversionRate'] * 60000
     df_marketing['ROI'] = (df_marketing['Revenue'] - df_marketing['AdSpend']) / df_marketing['AdSpend']
     ```

### 2. Customer Lifetime Value (CLV)
   - CLV focuses on expected revenue from a customer over a set period, adjusted by loyalty points and past purchase behavior.
   - Formula:
     ```python
     df_marketing['CLV'] = (60000 + df_marketing['LoyaltyPoints']) * df_marketing['PreviousPurchases']
     ```

### 3. Conversion Rates
   - Conversion rates for each customer are recorded in the dataset.

### 4. Campaign Costs
   - Represented by `AdSpend`, this is a significant contributor to campaign ROI.

```python
avg_revenue_per_conversion = 60000
df_marketing['Revenue'] = df_marketing['ConversionRate'] * avg_revenue_per_conversion
df_marketing['ROI'] = (df_marketing['Revenue'] - df_marketing['AdSpend']) / df_marketing['AdSpend']
df_marketing['CLV'] = (avg_revenue_per_conversion + df_marketing['LoyaltyPoints']) * df_marketing['PreviousPurchases']
```

## Feature Selection
To identify which features most impact ROI, we used a Random Forest Regressor to compute feature importance.

```python
X = df_marketing.drop(columns=['ROI', 'CampaignChannel'])
y = df_marketing['ROI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)
print("Feature importances (sorted):")
print(feature_importances_sorted)

```

Below is the results for the most important features to predict ROI

```
Feature importances (sorted):
AdSpend              0.741362
Revenue              0.127274
ConversionRate       0.110763
Income               0.008516
LoyaltyPoints        0.002780
Age                  0.002495
CLV                  0.002034
PreviousPurchases    0.001966
EmailClicks          0.001271
Conversion           0.000772
Gender               0.000462
CampaignType         0.000306
```

Campaign cost in this case is the greatest contributor to how effective our campaign is, followed by the revenue and conversion rate.

## Model Training and Results

Post Feature selection, we will begin training the model using linear regression, random forest and gradient boosting model for each of our 5 marketing strategies to predict the ROI and performance.

### Linear Regression

```python
# Ensure the 'Models' directory exists
if not os.path.exists('Models'):
    os.makedirs('Models')

# Get the top 7 features so that we have CLV Inclusive
top_features = feature_importances.nlargest(7).index

marketing_strategies = df_marketing['CampaignChannel'].unique().tolist()

# Conduct analysis for each marketing strategy
for strategy in marketing_strategies:
    print(f"========== {strategy} ==========")

    # Filter dataset for the current strategy
    X_strategy = df_marketing[df_marketing['CampaignChannel'] == strategy]
    Y_strategy = X_strategy['ROI']
    X_strategy = X_strategy.drop(columns=['ROI', 'CampaignChannel'])
    X_strategy = X_strategy[top_features]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_strategy, Y_strategy, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R2 Score: {r2}")
    print(f'Predicted ROI: {np.mean(y_pred)}')
    print()

    # Save the model
    model_filename = f"Models/linear_regression_model_{strategy}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

```

Results for linear regression below:

```
========== Email ==========
R2 Score: 0.3434977186033117
Predicted ROI: 1.6946128351597802
========== PPC ==========
R2 Score: 0.32562338597022045
Predicted ROI: 1.3330641359771054
========== Social Media ==========
R2 Score: 0.27380509728666835
Predicted ROI: 2.4489878903419156
========== Referral ==========
R2 Score: 0.2836530701122941
Predicted ROI: 1.8576880607150315
========== SEO ==========
R2 Score: 0.36805139997725345
Predicted ROI: 1.9171140411773933
```


### Random Forest
Results for random forest below:

========== Email ==========
Mean Absolute Error (MAE): 0.3635846723121918
Root Mean Squared Error (RMSE): 1.4621629173026365
R2 Score: 0.8999945739883014
ROI Mean: 1.4581831090204065

========== PPC ==========
Mean Absolute Error (MAE): 0.13090542932784355
Root Mean Squared Error (RMSE): 0.3614253063283053
R2 Score: 0.990498427443445
ROI Mean: 1.8895801537151282

========== Social Media ==========
Mean Absolute Error (MAE): 0.517514965653648
Root Mean Squared Error (RMSE): 2.1296559773232175
R2 Score: 0.8633737953258844
ROI Mean: 2.1971547087470658

========== Referral ==========
Mean Absolute Error (MAE): 0.7038192118197603
Root Mean Squared Error (RMSE): 3.1500072996704787
R2 Score: 0.7735972963734217
ROI Mean: 1.8996245008144463

========== SEO ==========
Mean Absolute Error (MAE): 0.8148669522459936
Root Mean Squared Error (RMSE): 3.3971226004689283
R2 Score: 0.6674107186663398
ROI Mean: 1.9732796784077806

# Gradient Boosting

Gradient boosting model results below:

========== Email ==========
Mean Absolute Error (MAE): 0.31036795282328883
Root Mean Squared Error (RMSE): 1.4629699982375177
R2 Score: 0.8998841420419087
ROI Mean: 1.4581831090204065

========== PPC ==========
Mean Absolute Error (MAE): 0.15835061863844893
Root Mean Squared Error (RMSE): 0.37407659944440647
R2 Score: 0.9898216013495315
ROI Mean: 1.8895801537151282

========== Social Media ==========
Mean Absolute Error (MAE): 0.8069433342812458
Root Mean Squared Error (RMSE): 3.7075425880202606
R2 Score: 0.5859170068090196
ROI Mean: 2.1971547087470658

========== Referral ==========
Mean Absolute Error (MAE): 0.37660525726015814
Root Mean Squared Error (RMSE): 1.236299045631206
R2 Score: 0.9651256954110944
ROI Mean: 1.8996245008144463

========== SEO ==========
Mean Absolute Error (MAE): 0.5236326716013449
Root Mean Squared Error (RMSE): 1.9131095551317019
R2 Score: 0.8945211250078701
ROI Mean: 1.9732796784077806

We will dynamically select the best model to predict the ROI for each strategy.


# Future Work

## Areas for Further Research
- Investigate new data sources for more detailed customer profiles.

## Potential Enhancements to the Current Solution
- Implement a real-time recommendation engine.

---

# Lessons Learned

## Challenges Faced and How They Were Overcome
- Difficulty in handling data inconsistencies, resolved with data preprocessing steps.

## Insights Gained During the Project
- Importance of feature engineering for model accuracy.

## Reflections on the Team's Approach and Process
- Effective collaboration between data science and marketing enhanced project success.

---

# References

Rathi, P. (2020). Banking Dataset Marketing Targets. Kaggle. Retrieved from https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets

Silva, J., & Santos, M. (2021). Enhancing Customer Engagement in Banking Through AI-Driven Marketing Strategies. *Journal of Banking & Finance*, 45(3), 112-125.

Tariq, H. (2022). Challenges and Opportunities in European Retail Banking. *European Financial Review*, 39(5), 87-93.

---

# Appendices

- **Appendix C: Synthetic Data Generation Notebook**
  - **Description**: Documents the process of generating synthetic product columns using a machine learning model. Includes the full methodology and code.
  - **Link**: [Link to `[product_synthetic_generation.ipynb](https://github.com/huangchaohung/Group-20/edit/main/project_wiki.md)`]
 
## Any Additional Information That Doesn't Fit into the Main Sections
- Extended data exploration summaries and additional analyses.

## Detailed Technical Explanations or Proofs, If Necessary
- Technical notes on feature engineering or advanced model tuning.
