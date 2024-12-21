import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('path_to_csv_file.csv')

# Handling missing data
data.fillna(method='ffill', inplace=True)  # Forward fill missing values
data.dropna(subset=['important_column'], inplace=True)  # Drop rows where 'important_column' is NaN

# Creating new features
data['new_feature'] = data['existing_column1'] * data['existing_column2']

# Merging datasets
other_data = pd.read_csv('path_to_other_csv_file.csv')
merged_data = pd.merge(data, other_data, on='common_column', how='inner')

# Pivoting data
pivot_table = data.pivot_table(values='value_column', index='index_column', columns='column_to_pivot', aggfunc=np.mean)

# Grouping and aggregating data
grouped_data = data.groupby('group_column').agg({
    'numeric_column1': 'sum',
    'numeric_column2': 'mean',
    'categorical_column': lambda x: x.mode()[0]
}).reset_index()

# Advanced filtering
filtered_data = data[(data['numeric_column'] > 50) & (data['categorical_column'] == 'specific_value')]

# Sorting data
sorted_data = data.sort_values(by=['numeric_column'], ascending=False)

# Applying custom functions
def custom_function(x):
    return x * 2

data['custom_feature'] = data['numeric_column'].apply(custom_function)

# Saving the processed data
data.to_csv('processed_data.csv', index=False)

# Example usage
print("First 5 rows of the processed data:")
print(data.head())
