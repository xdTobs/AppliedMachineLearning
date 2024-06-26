# Made by: Tobias Schønau s224327
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import dataloader

class ModelController():
    def __init__(self, model):
        self.model = model
        
    def predict(self, data, k_fold_split: int = 5):
        
        mse_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        negative_predictive_value_scores = []
        specificity_scores = []
        true_positive_scores = []
        true_negative_scores = []
        false_positive_scores = []
        false_negative_scores = []
        failed_mean_values = []
        successful_mean_values = []
        kf = KFold(n_splits=k_fold_split, shuffle=True)
        
        for train_index, test_index in kf.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]
            
            # We have low amount of class 1 samples, so we will upsample them in the training data
            class_1 = train[train['Class'] == 1]
            class_0 = train[train['Class'] == 0]
            class_1_upsampled = class_1.sample(n=len(class_0), replace=True, random_state=42)
            balanced_train : pd.DataFrame = pd.concat([class_0, class_1_upsampled])

            X_train, y_train = dataloader.split_variables_and_target(balanced_train)
            X_test, y_test = dataloader.split_variables_and_target(test)
                        
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
            
            true_negative = cm[0, 0]
            false_positive = cm[0, 1]
            false_negative = cm[1, 0]
            true_positive = cm[1, 1]

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            negative_predictive_value = true_negative / (true_negative + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            

            precision_scores.append(precision)
            recall_scores.append(recall)
            negative_predictive_value_scores.append(negative_predictive_value)
            specificity_scores.append(specificity)
            true_positive_scores.append(true_positive)
            true_negative_scores.append(true_negative)
            false_positive_scores.append(false_positive)
            false_negative_scores.append(false_negative)
            
            failed_predictions = X_test[y_test != classifier_predictions]
            successful_predictions = X_test[y_test == classifier_predictions]
            
            failed_mean_values.append(np.mean(failed_predictions["int.rate"]))
            successful_mean_values.append(np.mean(successful_predictions["int.rate"]))
            
            
            
            
        self.mse_avg = np.mean(mse_scores)
        self.accuracy_avg = np.mean(accuracy_scores)
        self.precision_avg = np.mean(precision_scores)
        self.recall_avg = np.mean(recall_scores)
        self.negative_predictive_value_avg = np.mean(negative_predictive_value_scores)
        self.specificity_avg = np.mean(specificity_scores)
        self.true_positive_avg = int(np.mean(true_positive_scores))
        self.true_negative_avg = int(np.mean(true_negative_scores))
        self.false_positive_avg = int(np.mean(false_positive_scores))
        self.false_negative_avg = int(np.mean(false_negative_scores))
        self.failed_mean_values = np.mean(failed_mean_values)
        self.successful_mean_values = np.mean(successful_mean_values)
        
