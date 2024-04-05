from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from functions import *
import joblib

folder_n = './Negative/'
folder_p = './Positive/'
width = 240
height = 240

df1 = preprocess_folder(folder_n, width, height)
df2 = preprocess_folder(folder_p, width, height)
df = pd.concat([df1, df2], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

joblib.dump(df, 'data.pkl')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(rf_classifier, 'random_forest_model.pkl')