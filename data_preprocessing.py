'''
Some data preprocessing to make the data usable for the nearest neighbors algorithm
Author: Zaid
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


dataset = pd.read_csv('thyroid_cancer_risk_data.csv')

label1 = dataset['Diagnosis']
label2 = dataset['Thyroid_Cancer_Risk']
labels = pd.concat([label2, label1], axis=1)
labels['Label'] = labels['Thyroid_Cancer_Risk'] + " " + labels['Diagnosis']

id = dataset[['Patient_ID']]

dataset = dataset.drop(['Diabetes','Patient_ID', 'Country', 'Ethnicity', 'Thyroid_Cancer_Risk', 'Diagnosis', 'T3_Level', "T4_Level"], axis=1)

le = LabelEncoder() 
labels['Label_encoded'] = le.fit_transform(labels['Label']) # Encoding the labels

# High Benign = 0
# High Malignant = 1
# Low Benign = 2
# Low Malignant = 3
# Medium Benign = 4
# Medium Malignant = 5

cols = ['Gender','Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity']
for col in cols:
    dataset[col] = le.fit_transform(dataset[col]) # encoding non-numeric binary data

# Gender --> Male = 1, Female = 0
# Rest --> yes = 1, no = 0

scaler = StandardScaler()
scaled_array = scaler.fit_transform(dataset) # Scale/normalize the dataset

# Assuming your scaler is named 'scaler'
print("Means:", list(scaler.mean_))
print("Stds:", list(scaler.scale_))

dataset = pd.DataFrame(scaled_array, columns=dataset.columns)

dataset.to_csv('features.csv', index=False)
labels.to_csv('labels.csv', index=False)
id.to_csv('ids.csv', index = False)



