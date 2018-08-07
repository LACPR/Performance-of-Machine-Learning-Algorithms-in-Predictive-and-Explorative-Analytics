#PCA - Dimensionality Reduction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset = pd.read_csv("diabetic_data.csv", na_values="?")

dataset.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# ICD9 Codes for diag_1, diag_2 and diag_3
def ICD9_classifier(diagnoses):
    # Returns a series of strings corresponding to type of ICD9 diagnosis
    # diagnoses is a list
    gr_diagnoses = diagnoses.copy()
    icd9_dct = {
                'Infectious':(1, 139),
                'Neoplasmic':(140,239),
                'Hormonal':(240, 279),
                'Blood':(280,289),
                'Mental':(290,319),
                'Nervous':(320,359),
                'Sensory':(360,389),
                'Circulatory':(390,459),
                'Respiratory':(460,519),
                'Digestive':(520,579),
                'Genitourinary':(580,629),
                'Childbirth':(630,679),
                'Dermatological':(680,709),
                'Musculoskeletal':(710,739),
                'Congenital':(740,759),
                'Perinatal':(760,779),
                'Miscellaneous':(780,799),
                'Injury':(800,999)
               }
    for i, diagnosis in enumerate(diagnoses):
        if (str(diagnoses[i])[0] == 'E') or (str(diagnoses[i])[0] == 'V'):
            gr_diagnoses[i] = 'Accidental'
        elif (str(diagnoses[i]).lower() == 'nan'):
            gr_diagnoses[i] = 'NaN'
        else:
            for key, icd_range in icd9_dct.items():
                if (int(float(diagnoses[i])) >= icd_range[0]) and (int(float(diagnoses[i])) <= icd_range[1]):
                    gr_diagnoses[i] = key
    return gr_diagnoses

d1 = ICD9_classifier(dataset.diag_1.values)
d2 = ICD9_classifier(dataset.diag_2.values)
d3 = ICD9_classifier(dataset.diag_3.values)

dataset.diag_1 = d1
dataset.diag_2 = d2
dataset.diag_3 = d3

finalDataset = dataset.dropna()

# Label-encode age feature to an integer in the center of the raw bin
finalDataset.age = (LabelEncoder().fit_transform(finalDataset.age)*10) + 5

# Categorical data should be encoded
lb = LabelEncoder()
finalDataset['race'] = lb.fit_transform(finalDataset['race'])
finalDataset['gender'] = lb.fit_transform(finalDataset['gender'])
finalDataset['diag_1'] = lb.fit_transform(finalDataset['diag_1']) 
finalDataset['diag_2'] = lb.fit_transform(finalDataset['diag_2'])
finalDataset['diag_3'] = lb.fit_transform(finalDataset['diag_3'])
finalDataset['max_glu_serum'] = lb.fit_transform(finalDataset['max_glu_serum']) 
finalDataset['A1Cresult'] = lb.fit_transform(finalDataset['A1Cresult']) 
finalDataset['metformin'] = lb.fit_transform(finalDataset['metformin'])
finalDataset['repaglinide'] = lb.fit_transform(finalDataset['repaglinide'])
finalDataset['nateglinide'] = lb.fit_transform(finalDataset['nateglinide']) 
finalDataset['chlorpropamide'] = lb.fit_transform(finalDataset['chlorpropamide']) 
finalDataset['glimepiride'] = lb.fit_transform(finalDataset['glimepiride'])
finalDataset['acetohexamide'] = lb.fit_transform(finalDataset['acetohexamide'])
finalDataset['glipizide'] = lb.fit_transform(finalDataset['glipizide']) 
finalDataset['glyburide'] = lb.fit_transform(finalDataset['glyburide']) 
finalDataset['tolbutamide'] = lb.fit_transform(finalDataset['tolbutamide'])
finalDataset['pioglitazone'] = lb.fit_transform(finalDataset['pioglitazone'])
finalDataset['rosiglitazone'] = lb.fit_transform(finalDataset['rosiglitazone']) 
finalDataset['acarbose'] = lb.fit_transform(finalDataset['acarbose']) 
finalDataset['miglitol'] = lb.fit_transform(finalDataset['miglitol'])
finalDataset['troglitazone'] = lb.fit_transform(finalDataset['troglitazone'])
finalDataset['tolazamide'] = lb.fit_transform(finalDataset['tolazamide']) 
finalDataset['examide'] = lb.fit_transform(finalDataset['examide']) 
finalDataset['citoglipton'] = lb.fit_transform(finalDataset['citoglipton'])
finalDataset['insulin'] = lb.fit_transform(finalDataset['insulin'])
finalDataset['glyburide_metformin'] = lb.fit_transform(finalDataset['glyburide_metformin']) 
finalDataset['glipizide_metformin'] = lb.fit_transform(finalDataset['glipizide_metformin']) 
finalDataset['glimepiride_pioglitazone'] = lb.fit_transform(finalDataset['glimepiride_pioglitazone'])
finalDataset['metformin_rosiglitazone'] = lb.fit_transform(finalDataset['metformin_rosiglitazone'])
finalDataset['metformin_pioglitazone'] = lb.fit_transform(finalDataset['metformin_pioglitazone']) 
finalDataset['change'] = lb.fit_transform(finalDataset['change'])
finalDataset['diabetesMed'] = lb.fit_transform(finalDataset['diabetesMed']) 
finalDataset['readmitted'] = lb.fit_transform(finalDataset['readmitted'])

X = finalDataset.iloc[:, :-1].values
y = finalDataset.iloc[:, 44].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

