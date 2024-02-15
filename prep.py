import pandas as pd

def prep():
    # get data
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # drop 'SalePrice'
    X = pd.concat([train_data.drop(columns=['SalePrice']),test_data],ignore_index=True)
    #calculate the percentage of null values in the columns
    null_percent = X.isnull().sum()/X.shape[0]*100
    # deleting the columns with more than 50 missing values
    col_to_drop = null_percent[null_percent > 50].keys()
    X = X.drop(columns=list(col_to_drop))
    # feature engineering
    numerical_cols = X.loc[:, X.isnull().any()].select_dtypes(include='number').columns
    categorical_cols = X.loc[:, X.isnull().any()].select_dtypes(exclude='number').columns
    print("# Numerical columns with null values:", len(numerical_cols))
    print("# Categorical columns with null values:", len(categorical_cols))
    for column in numerical_cols:
        # Replace missing values with the mean
        X[column] = X[column].fillna(X[column].mean())
    for column in categorical_cols:
        # Replace missing values with the mode
        X[column] = X[column].fillna(X[column].mode()[0])
    if not X.isnull().values.any():
        print("\nThere are no missing values.")
    # One-hot encoding
    X = pd.get_dummies(data=X)
    # export
    test_data_transform = X.iloc[train_data.shape[0]:].copy()
    X = X.iloc[:train_data.shape[0]].copy()
    test_data_transform.to_parquet('./data/test_prep.parquet')
    X.to_parquet('./data/train_prep.parquet')

if __name__ == '__main__':
    prep()