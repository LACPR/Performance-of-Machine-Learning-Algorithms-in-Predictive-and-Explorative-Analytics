# Major Libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)

dataset.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

dataset = dataset.dropna()

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

dataset.age = (LabelEncoder().fit_transform(dataset.age)*10) + 5

lb = LabelEncoder()
dataset['race'] = lb.fit_transform(dataset['race'])
dataset['gender'] = lb.fit_transform(dataset['gender']) 
dataset['diag_1'] = lb.fit_transform(dataset['diag_1']) 
dataset['diag_2'] = lb.fit_transform(dataset['diag_2'])
dataset['diag_3'] = lb.fit_transform(dataset['diag_3'])
dataset['max_glu_serum'] = lb.fit_transform(dataset['max_glu_serum']) 
dataset['A1Cresult'] = lb.fit_transform(dataset['A1Cresult']) 
dataset['metformin'] = lb.fit_transform(dataset['metformin'])
dataset['repaglinide'] = lb.fit_transform(dataset['repaglinide'])
dataset['nateglinide'] = lb.fit_transform(dataset['nateglinide']) 
dataset['chlorpropamide'] = lb.fit_transform(dataset['chlorpropamide']) 
dataset['glimepiride'] = lb.fit_transform(dataset['glimepiride'])
dataset['acetohexamide'] = lb.fit_transform(dataset['acetohexamide'])
dataset['glipizide'] = lb.fit_transform(dataset['glipizide']) 
dataset['glyburide'] = lb.fit_transform(dataset['glyburide']) 
dataset['tolbutamide'] = lb.fit_transform(dataset['tolbutamide'])
dataset['pioglitazone'] = lb.fit_transform(dataset['pioglitazone'])
dataset['rosiglitazone'] = lb.fit_transform(dataset['rosiglitazone']) 
dataset['acarbose'] = lb.fit_transform(dataset['acarbose']) 
dataset['miglitol'] = lb.fit_transform(dataset['miglitol'])
dataset['troglitazone'] = lb.fit_transform(dataset['troglitazone'])
dataset['tolazamide'] = lb.fit_transform(dataset['tolazamide']) 
dataset['examide'] = lb.fit_transform(dataset['examide']) 
dataset['citoglipton'] = lb.fit_transform(dataset['citoglipton'])
dataset['insulin'] = lb.fit_transform(dataset['insulin'])
dataset['glyburide_metformin'] = lb.fit_transform(dataset['glyburide_metformin']) 
dataset['glipizide_metformin'] = lb.fit_transform(dataset['glipizide_metformin']) 
dataset['glimepiride_pioglitazone'] = lb.fit_transform(dataset['glimepiride_pioglitazone'])
dataset['metformin_rosiglitazone'] = lb.fit_transform(dataset['metformin_rosiglitazone'])
dataset['metformin_pioglitazone'] = lb.fit_transform(dataset['metformin_pioglitazone']) 
dataset['change'] = lb.fit_transform(dataset['change'])
dataset['diabetesMed'] = lb.fit_transform(dataset['diabetesMed']) 
dataset['readmitted'] = lb.fit_transform(dataset['readmitted'])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])
dataset = onehotencoder.fit_transform(dataset).toarray()

X = dataset[:, :156]
y = dataset[:, 156:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_