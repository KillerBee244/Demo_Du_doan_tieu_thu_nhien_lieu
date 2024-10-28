from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('auto_mpg.csv')


df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')


df = df.dropna()


df = df.drop(columns=['car name'])


X = df.drop(columns=['mpg'])  
y = df['mpg'] 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return r2, rmse, mae, nse

model_linear = joblib.load('linear_model.pkl')
model_Mlp = joblib.load('Mlp_model.pkl')
model_lasso = joblib.load(lasso_model.pkl')
model_stacking = joblib.load('stacking_model.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input from form
        cylinders = float(request.form["cylinders"])
        displacement = float(request.form["displacement"])
        horsepower = float(request.form["horsepower"])
        weight = float(request.form["weight"])
        acceleration = float(request.form["acceleration"])
        model_year = float(request.form["model_year"])
        origin = float(request.form["origin"])
        model_type = request.form["model"]
        
        # Prepare input for prediction
        input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])

        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        
        # Select model
        if model_type == "linear_regression":
            model = model_linear
        elif model_type == "lasso":
            model =   model_lasso # Adjust alpha for Lasso regularization strength
        elif model_type == "mlp":
            model = model_Mlp
        elif model_type == "stk":
            model = model_stacking

        
        # Train model
        y_pred_train = model.predict(X_train)
        
        # Make prediction
        y_pred = model.predict(input_data_scaled)
        
        # Calculate performance metrics
        r2, rmse, mae, nse = calculate_metrics(y_train, y_pred_train)
        
        # Render result without plot
        return render_template('result.html', y_pred=y_pred[0], r2=r2, rmse=rmse, mae=mae, nse=nse)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
