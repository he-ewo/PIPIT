from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import lightgbm as lgb

def print_metrics(y_real, y_pred):
    accuracy = round(accuracy_score(y_real, y_pred), 4)
    precision = round(precision_score(y_real, y_pred, average='weighted'), 4)
    recall = round(recall_score(y_real, y_pred, average='weighted'), 4)
    f1 = round(f1_score(y_real, y_pred, average='weighted'), 4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)



def RandomForest(x_train, y_train, x_test, y_test):
    # 모델링
    model_rf = RandomForestClassifier(n_estimators=15, random_state=42)
    
    # 학습
    model_rf.fit(x_train, y_train)

    # 예측
    y_pred = model_rf.predict(x_test) 

    # 평가
    print_metrics(y_test, y_pred)

def LGBM(x_train, y_train, x_test, y_test):
    # 모델링
    model_lgbm = lgb.LGBMClassifier(n_estimators=15, force_col_wise=True)
    # 학습
    model_lgbm.fit(x_train, y_train)

    # 예측
    y_pred = model_lgbm.predict(x_test) 

    # 평가
    print_metrics(y_test, y_pred)
