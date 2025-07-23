
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Training.csv')
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

symptom_columns = list(X.columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    return jsonify({'symptoms': symptom_columns})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['symptoms']
    input_data = [0] * len(symptom_columns)
    for symptom in data:
        if symptom in symptom_columns:
            input_data[symptom_columns.index(symptom)] = 1
    prediction = model.predict([input_data])
    result = le.inverse_transform(prediction)[0]
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
