import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

class DataLoaderInterface:
    # Interface for data loaders
    # Data should be loaded in the __init__ method
    # Data should be standardized in the __init__ method
    
    # Data should have a column named 'Class' which is the target variable
    def __init__(self):
        pass
    def get_data(self) -> pd.DataFrame:
        pass

    
class LoanForest(DataLoaderInterface):
    def __init__(self):
        self.url = 'data/loan_data.csv'
        self.column_names = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'Class']
        self.data : pd.DataFrame = pd.read_csv(self.url, names =self.column_names, na_values="?")
        self.data = self.data.dropna()
        # Encode string columns using LabelEncoder
        string_columns = ['purpose']
        for column in string_columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
        
        
    def get_data(self) -> pd.DataFrame:
        return self.data

class LoanNeural(DataLoaderInterface):
    def __init__(self):
        self.url = 'data/loan_data.csv'
        self.column_names = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'Class']
        self.data : pd.DataFrame = pd.read_csv(self.url, names =self.column_names, na_values="?")
        self.data = self.data.dropna()
        
        # Label encode the purpose column using one hot encoding
        self.data = pd.get_dummies(self.data, columns=['purpose'])
        
        # We have low amount of class 1 samples, so we will upsample them
        class_1 = self.data[self.data['Class'] == 1]
        class_0 = self.data[self.data['Class'] == 0]

        class_1_upsampled = class_1.sample(n=len(class_0), replace
        =True, random_state=42)
        self.balanced_data : pd.DataFrame = pd.concat([class_0, class_1_upsampled])
    
    def get_data(self) -> pd.DataFrame:
        return self.balanced_data
    
    def scale_data(self, data : pd.DataFrame):
        self.scaler = StandardScaler()
        data.iloc[:, 1:] = self.scaler.fit_transform(data.iloc[:, 1:])
        return data
    

class CreditCard(DataLoaderInterface):
    def __init__(self):
        self.url = 'data/creditcard.csv'
        self.column_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
        self.data : pd.DataFrame = pd.read_csv(self.url, names=self.column_names, na_values="?")
        self.data = self.data.dropna()

        self.data['Class'] = pd.to_numeric(self.data['Class'], errors='coerce')

        # Standardize the data
        self.scaler = StandardScaler()
        self.data.iloc[:, :-1] = self.scaler.fit_transform(self.data.iloc[:, :-1])


        # We have low amount of class 1 samples, so we will upsample them
        class_1 = self.data[self.data['Class'] == 1]
        class_0 = self.data[self.data['Class'] == 0]

        class_1_upsampled = class_1.sample(n=len(class_0), replace
        =True, random_state=42)
        self.balanced_data : pd.DataFrame = pd.concat([class_0, class_1_upsampled])
        
    def get_raw_data(self) -> pd.DataFrame:
        return self.data
    
    def get_balanced_data(self) -> pd.DataFrame:
        return self.balanced_data
        

            
            
def split_variables_and_target(data : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = data.drop(['Class'], axis=1)
        y = data['Class']
        return X, y

def split_data(data : pd.DataFrame, test_size : float = 0.2) -> list:
        X, y = split_variables_and_target(data)
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
def get_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
