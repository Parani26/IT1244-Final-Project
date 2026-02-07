import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Read the CSV file into a DataFrame
train_data = pd.read_csv('Train_Set.csv')
train_data.head(5)
# 2193 rows and 351 columns
print(train_data.info())
print(train_data.shape)
train_data.describe()
train_data.isna().sum()
# no missing data points
print(train_data.iloc[:, -1])
print(train_data['class_label'].value_counts())
### class_label
# early stage cancer        781
# screening stage cancer    490
# mid stage cancer          453
# late stage cancer         409
# healthy                    60
label_encoder = LabelEncoder()
train_data['class_label'] = label_encoder.fit_transform(train_data['class_label'])
train_data["class_label"].unique()
### class_label
# early stage cancer --> 0       781
# screening stage cancer  --> 4  490
# mid stage cancer  --> 3        453
# late stage cancer  --> 2       409
# healthy  --> 1                  60
import matplotlib.pyplot as plt
# use of a correlation matrix to observe how variables in our dataset may be affected by others
numerical_train_data = train_data.select_dtypes(include=['float64', 'int64'])

# Create a correlation matrix
correlation_matrix = numerical_train_data.corr()

# Assuming correlation_matrix is your correlation matrix
correlation_df = pd.DataFrame(correlation_matrix)

# Print or use the correlation DataFrame as needed
correlation_df