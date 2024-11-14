import pandas as pd
import numpy as np

#function to generate synthetic transaction data using common feature columns
def find_and_assign_values(row, df_combine):
    # Conditions for finding matching rows: Age within ±5 and Balance within ±10%
    condition = (
        (df_combine['Age'] >= row['age'] - 15) & (df_combine['Age'] <= row['age'] + 15) &
        (df_combine['CustAccountBalance'] >= row['balance'] * 0.7 - 100) & (df_combine['CustAccountBalance'] <= max(100, row['balance'] * 1.3 + 100))
    )
    
    # Get matching rows from df_combine
    matching_rows = df_combine[condition]
    
    # If a match is found, assign the closest one and remove it from df_combine
    closest_match = matching_rows.iloc[0]
    
    # Assign the new columns to the row in df
    row['TransactionFrequency'] = closest_match['TransactionFrequency']
    row['Recency'] = closest_match['Recency']
    row['AverageTransactionAmount'] = closest_match['AverageTransactionAmount']
    
    # Remove the matched row from df_combine
    df_combine.drop(closest_match.name, inplace=True)
    print(row)
    
    return row

#import the dataset without transaction information
df = pd.read_csv(r"../data/clean_dataset.csv")
df.drop(columns=['day', 'month'], inplace=True)
df['housing'] = df['housing'].replace({'yes': 1, 'no': 0})
df['loan'] = df['loan'].replace({'yes': 1, 'no': 0})

#import the transaction data from another dataset from kaggle to generate synthetic data
df_combine = pd.read_csv(r"../data/Combine.csv")

#create synthetic data
df = df.apply(find_and_assign_values, axis=1, df_combine=df_combine)

#output the dataset for later use
df.to_csv('Combined_dataset.csv', index=False)