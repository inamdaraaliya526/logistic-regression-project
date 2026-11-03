# Titanic Survival Prediction using Logistic Regression

## Objective
Predict whether a Titanic passenger survived using passenger class, age, sex, and fare.

## Dataset
Download `train.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data).
Place it in the same folder as `logistic_regression.py`.

## Method
1. Select passenger class, age, sex, and fare as input features.
2. Remove rows with missing values in those columns.
3. Convert the sex category into numeric data.
4. Split the data into training and test sets.
5. Scale the features and train a logistic regression model.
6. Evaluate the model using accuracy and a classification report.

## Run
Install Python, then run:

```bash
python -m pip install -r requirements.txt
python logistic_regression.py
```

## Result
Run the script and write the accuracy shown in your output here.

## Conclusion
The model uses passenger details to estimate survival. Its test score is an estimate and may differ on other data.
