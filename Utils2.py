import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
def weekend_filter(df):
    adj = []
    for i in df["Weekend"]:
        if i == 1:
            adj.append(int(1))
        elif i==7:
            adj.append(int(1))
        else:
            adj.append(int(0)) 
    df["Weekend"] = adj
    return(df)

def Scaler(df):    
    a = list(df["hnt_Mined"].reset_index(drop=True))
    b = list(df["tmax"].reset_index(drop=True))
    scaler = MinMaxScaler()
    mining_shaped = np.reshape(a,(-1,1))
    tempature_shaped = np.reshape(b,(-1,1))
    mining_scaled= scaler.fit_transform(mining_shaped)
    tempature_scaled = scaler.fit_transform(tempature_shaped)
    return mining_scaled, tempature_scaled

def ind_ttest(High,Low,alpha):
    t, pval =stats.ttest_ind(High,Low)
    print("t:", t, "pval:", pval, "alpha:", alpha)
    if pval < alpha:
        print("reject H0")
    else: 
        print("do not reject H0")

def convert_value_mined(df):
    vmdf = df["Mined_Value"]
    cat = []
    for i in vmdf:
        if i > np.median(vmdf):
            cat.append(1)
        else:
            cat.append(0)
    df["Mined_Value"]=cat
    return df

def decision_tree_classifier(df,class_collumn):
    y = list(df[class_collumn]) #seperate class data
    X = df.drop([class_collumn,"hnt_Mined"], axis = 1) #drop class data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y) #split
    tree_clf = DecisionTreeClassifier(random_state=0) #select classifier
    tree_clf.fit(X_train,y_train) #fit classifier
    y_predicted = tree_clf.predict(X_test) #predict class values for test data
    accuracy = accuracy_score(y_test,y_predicted) #measure accuracy
    print("Decision Tree Classifier Accuracy:", round(accuracy,2))
    return accuracy

def KNN_classifier(df,class_collumn):
    y = list(df[class_collumn]) #seperate class data
    X = df.drop([class_collumn,"hnt_Mined"], axis = 1) #drop class data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X) #standardize feature data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)#split/train
    knn_clf = KNeighborsClassifier(n_neighbors=9) #define classifier
    knn_clf.fit(X_train, y_train) #fit classifier
    #predict
    y_predicted = knn_clf.predict(X_test) #recieve predictions
    metrics = accuracy_score(y_test, y_predicted) 
    print("KNN Classifier Accuracy", round(metrics,2))
    return metrics
