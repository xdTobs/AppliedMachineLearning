# Welcome to my code

## Dependencies

All python requirements *Should* be in requirements.txt, and can be downloaded however you please: (```pip install -r requirements.txt```)

## Models

My models are located in the following files:

- logistic_regression.ipynb
- neural_network.ipynb
- support_vector_machine.ipynb
- ensemble.ipynb

## Data

All of these models get their data from the dataloader.py class, which is responsible for the preprocessing of the data. Some models use the modelController.py file to run their models. This class is responsible for using K-Fold, along with collecting stats used for graphs.

Data can be found in data/loan_data.csv

## Graphs

Graphs are made in the following files:

- dataset_overview.ipynb
- shap.ipynb
- ensemble.ipynb
- model_comparisons.ipynb

## GPT use

Chat-GPT has been used only for creating graphs. Here it has been very helpful, but it has not been used in other places.
