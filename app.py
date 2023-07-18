from flask import Flask, render_template, request, redirect, url_for
import csv
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.model_selection import train_test_split

app = Flask(__name__)
model_path = "rf_model.pkl"  
model_path1 = "model.pkl"
encoded_data_path = "csv_files/encoded.csv"
encoded_data_path1 = "csv_files/yencoded.csv"  
"""rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor1 = RandomForestRegressor(n_estimators=120, random_state=47)
encoded_data = pd.read_csv(encoded_data_path)
encoded_data1 = pd.read_csv(encoded_data_path1)
X = encoded_data.iloc[:, 6:].values
X1 = encoded_data1.iloc[:, 7:].values
y_subpopulation = encoded_data['subpopulation'].values
y_height = encoded_data['mean_height'].values
y_yield = encoded_data1['Yield'].values
X_train, X_test, y_subpopulation_train, y_subpopulation_test, y_height_train, y_height_test = train_test_split(
    X, y_subpopulation, y_height, test_size=0.2, random_state=42)
X1_train, X1_test, y_yield_train, y_yield_test = train_test_split(X1, y_yield, test_size=0.2, random_state=42)
rf_classifier.fit(X_train, y_subpopulation_train)
rf_regressor.fit(X_train, y_height_train)
rf_regressor1.fit(X1_train, y_yield_train)"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('predict.html')
    return render_template('index.html')

@app.route('/phylotree')
def phylotree():
    return render_template('phylo.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        dna_sequences = request.form.get('dna-sequences')
        predicted_subpopulation, predicted_height,predicted_yield = make_prediction(dna_sequences)
        return redirect(url_for('result', predicted_subpopulation=predicted_subpopulation, predicted_height=predicted_height,predicted_yield=predicted_yield))
    return render_template('predict.html')


@app.route('/predictions', methods=['GET', 'POST'])
def height():
    if request.method == 'POST':
        dna_sequences = request.form.get('dna-sequences')
        predicted_subpopulation, predicted_height, predicted_yield = make_prediction(dna_sequences)
        return redirect(url_for('result', predicted_subpopulation=predicted_subpopulation, predicted_height=predicted_height,predicted_yield=predicted_yield))
    return render_template('predictions.html')

@app.route('/result', methods=['GET'])
def result():
    predicted_subpopulation = request.args.get('predicted_subpopulation')
    predicted_height = request.args.get('predicted_height')
    predicted_yield = request.args.get('predicted_yield')
    return render_template('result.html', predicted_subpopulation=predicted_subpopulation, predicted_height=predicted_height,predicted_yield=predicted_yield)

@app.route('/height-info')
def height_info():
    file_path = 'C:/Users/Prerana/Desktop/AGROGENOMICS/flask/data/heightinfo.csv'

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    return render_template('height_info.html', data=data)

@app.route('/yield-info')
def yield_info():
    file_path = 'C:/Users/Prerana/Desktop/AGROGENOMICS/flask/data/yieldinfo.csv'

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    return render_template('yield_info.html', data=data)

@app.route('/GWAS-file')
def gwas():
    file_path = 'C:/Users/Prerana/Desktop/AGROGENOMICS/flask/data/GWAS_file.csv'

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    return render_template('GWAS.html', data=data)



def make_prediction(dna_sequence):
    d = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    dna_sequence = dna_sequence.replace('\r', '')
    new_seq_encoded = np.array([d[nucleotide] for nucleotide in dna_sequence])
    new_seq_encoded = new_seq_encoded.reshape(1, -1)
    predicted_subpopulation = rf_classifier.predict(new_seq_encoded)[0]
    predicted_height = rf_regressor.predict(new_seq_encoded)[0]
    predicted_yield = rf_regressor1.predict(new_seq_encoded)[0]
    return predicted_subpopulation, predicted_height, predicted_yield

if __name__ == '__main__':
    app.run(debug=True)

