import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualization(df):
    # Visualize the distribution of 'Transaction Frequency' across different clusters
    plt.figure()
    sns.boxplot(x='cluster', y='TransactionFrequency', data=df)
    plt.title('Transaction Frequency Distribution across Clusters')
    plt.savefig('../../image/Transaction_Frequency.png', bbox_inches='tight')
    plt.close()

    # Visualize the distribution of 'Recency' across different clusters
    plt.figure()
    sns.boxplot(x='cluster', y='Recency', data=df)
    plt.title('Recency Distribution across Clusters')
    plt.savefig('../../image/Recency.png', bbox_inches='tight')
    plt.close()

    # Visualize the average 'Average Transaction Amount' for each cluster
    plt.figure()
    sns.barplot(data=df, x='cluster', y='AverageTransactionAmount', errorbar=None)
    plt.title('Average Transaction Amount by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Transaction Amount')
    plt.savefig('../../image/Transaction_Amount.png', bbox_inches='tight')
    plt.close()

    # Visualise the products of each segmented groups
    x = ['0', '1', '2', '3', '4', '5', '6']
    y1 = df.groupby('cluster')['loan'].mean()
    y2 = df.groupby('cluster')['housing'].mean()
    y3 = df.groupby('cluster')['cd_account'].mean()
    y4 = df.groupby('cluster')['securities'].mean()
    
    # plot bars in stack manner
    plt.figure()
    plt.bar(x, y1, color='r')
    plt.bar(x, y2, bottom=y1, color='b')
    plt.bar(x, y3, bottom=y1+y2, color='y')
    plt.bar(x, y4, bottom=y1+y2+y3, color='g')
    plt.xlabel("Segment")
    plt.ylabel("Score")
    plt.legend(['personal loan', 'housing loan', 'cd account', 'security account'])
    plt.title("Product ownership by all segmented groups")
    plt.savefig('../../image/Product_ownership.png', bbox_inches='tight')
    plt.close()
