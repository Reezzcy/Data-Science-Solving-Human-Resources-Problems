import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load('model/kmeans_clustering_model.joblib')
scaler = joblib.load('model/scaler.joblib')

def encoding_data(df):
    transAtribut = ["BusinessTravel", "Department", "EducationField", "JobRole", "OverTime"]
    for trans in transAtribut:
        labelEncoding = joblib.load(f'model/labelEncoding_{trans}.joblib')
        df[trans] = labelEncoding.transform(df[trans])

    return df

data = {
    'Age': [35, 40, 50],
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Research & Development', 'Research & Development', 'Sales'],
    'DistanceFromHome': [5, 10, 15],
    'Education': [3, 2, 4],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing'],
    'EnvironmentSatisfaction': [3, 2, 4],
    'JobInvolvement': [2, 3, 4],
    'JobLevel': [2, 2, 3],
    'JobRole': ['Research Scientist', 'Research Director', 'Sales Executive'],
    'JobSatisfaction': [4, 3, 2],
    'MonthlyIncome':[6230, 13237, 6172],
    'MonthlyRate': [13430, 20978, 10877],
    'NumCompaniesWorked': [3, 2, 7],
    'OverTime': ['No', 'Yes', 'No'],
    'PercentSalaryHike': [10, 17, 13],
    'PerformanceRating': [3, 4, 3],
    'RelationshipSatisfaction': [4, 3, 3],
    'StockOptionLevel': [0, 3, 1],
    'TotalWorkingYears': [16, 19, 14],
    'WorkLifeBalance': [3, 4, 2],
    'YearsAtCompany': [5, 10, 15],
    'YearsInCurrentRole': [2, 5, 8],
    'YearsSinceLastPromotion': [1, 3, 4],
    'YearsWithCurrManager': [2, 3, 4]
}

df = pd.DataFrame(data)

df_encoded = encoding_data(df)

df_scaled = scaler.transform(df_encoded)

predictions = model.predict(df_scaled)
predictions = predictions.tolist()

for i in predictions:
    print("Potentially Attrition" if i == 0 else "No Attrition")