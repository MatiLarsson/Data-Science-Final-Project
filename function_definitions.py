import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def createdummies(df, varnames):
  for var in varnames:
    dummy = pd.get_dummies(df[var], prefix = var, drop_first=True)
    df = df.drop(var, axis = 1)
    df = pd.concat([df, dummy], axis = 1)
  return df

def get_accuracy(X_train, X_test, y_train, y_test, model):
    return  {"test Accuracy":accuracy_score(y_test, model.predict(X_test)),"train Accuracy": accuracy_score(y_train, model.predict(X_train))}

def measure_errors(y_true, y_pred, y_probs, label):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred),
                     'auc': roc_auc_score(y_true,y_probs)},
                      name=label)

def measure_error_depth(y_true, y_pred, label, depth):
    return pd.Series({depth:accuracy_score(y_true, y_pred)},
                      name=label)