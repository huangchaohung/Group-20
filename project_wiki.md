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
- **Banking Dataset - Marketing Targets**: (description) https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets
- **Bank Personal Loan**: (description) https://www.kaggle.com/datasets/mahnazarjmand/bank-personal-loan/data
- **Customer Comments**: (description)

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
- Step-by-step guide for deploying the model using Docker.

## Monitoring and Maintenance Considerations
- Regular retraining, logging of model predictions, and real-time performance monitoring.

---

# Technical Implementation

## Repository Structure
- Organized by data processing, modeling, evaluation, and deployment folders.

## Setup Instructions
- Steps for installing dependencies and configuring the project.

## Dependency Management
- Requirements file listing dependencies with versions.

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
- Question 3:
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
