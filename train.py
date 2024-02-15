import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def train():
    # import data
    test_data_transform = pd.read_parquet('./data/test_prep.parquet')
    X = pd.read_parquet('./data/train_prep.parquet')
    train_data = pd.read_csv('./data/train.csv')
    # adjust columns
    X = X.drop(columns=['Id'])
    y = train_data['SalePrice']
    # Log-transform the target variable
    y_log = np.log(y)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Build the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    rf_model.fit(X_train_scaled, y_train)
    


# Save the scaler
dump(scaler, 'scaler.joblib')

# Save the model
dump(model, 'model.joblib')