from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("framingham.csv")
data.drop(columns=['education'], inplace=True)
data=data[data['cigsPerDay'].notna()]
data=data[data['BPMeds'].notna()]
data=data[data['totChol'].notna()]
data=data[data['BMI'].notna()]
data=data[data['heartRate'].notna()]
mean = round(data['glucose'].mean())
data = data.fillna({'glucose':mean})
data.head(5)
data=data.astype(int)
y = np.array(data['TenYearCHD'])
data.drop(columns=['TenYearCHD'], inplace=True)
x = np.array(data)
Scalar = StandardScaler()
x= Scalar.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
joblib.dump(model, 'model.pkl')
model1=joblib.load('model.pkl')
joblib.dump(Scalar, 'scalar.pkl')

