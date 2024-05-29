from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np

class ModelController():
    def __init__(self, model):
        self.model = model
        
    def predict(self, X, y, k_fold_split: int = 5):
        
        mse_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        true_positive_scores = []
        true_negative_scores = []
        failed_mean_values = []
        successful_mean_values = []
        kf = KFold(n_splits=k_fold_split, shuffle=True)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Train a random forest classifier
            self.model.fit(X_train, y_train)
            # Test on the X_test data
            classifier_predictions = self.model.predict(X_test)

            # Calculate the mean squared error
            mse = mean_squared_error(y_test, classifier_predictions)
            mse_scores.append(mse)
            #print("Random Forest Classifier MSE:", mse)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, classifier_predictions.round())
            accuracy_scores.append(accuracy)
            #print("Random Forest Classifier Accuracy:", accuracy)

            cm = confusion_matrix(y_test, classifier_predictions)

            true_positive = cm[0, 0]
            true_negative = cm[1, 1]
            false_positive = cm[0, 1]
            false_negative = cm[1, 0]


            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            precision_scores.append(precision)
            recall_scores.append(recall)
            true_positive_scores.append(true_positive / (true_positive + false_positive))
            true_negative_scores.append(true_negative / (true_negative + false_negative))
            
            failed_predictions = X_test[y_test != classifier_predictions]
            successful_predictions = X_test[y_test == classifier_predictions]
            
            failed_mean_values.append(np.mean(failed_predictions["int.rate"]))
            successful_mean_values.append(np.mean(successful_predictions["int.rate"]))
            
            
            
            
        self.mse_avg = np.mean(mse_scores)
        self.accuracy_avg = np.mean(accuracy_scores)
        self.precision_avg = np.mean(precision_scores)
        self.recall_avg = np.mean(recall_scores)
        self.true_positive_avg = np.mean(true_positive_scores)
        self.true_negative_avg = np.mean(true_negative_scores)
        self.failed_mean_values = np.mean(failed_mean_values)
        self.successful_mean_values = np.mean(successful_mean_values)
        
