import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    a = list(df["hnt_Mined"].reset_index(drop=True))
    b = list(df["tmax"].reset_index(drop=True))
    scaler = MinMaxScaler()
    mining_shaped = np.reshape(a,(-1,1))
    tempature_shaped = np.reshape(b,(-1,1))
    mining_scaled= scaler.fit_transform(mining_shaped)
    tempature_scaled = scaler.fit_transform(tempature_shaped)
    return mining_scaled, tempature_scaled

def ind_ttest(High,Low,alpha):
    from scipy import stats
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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    y = list(df[class_collumn]) #seperate class data
    X = df.drop([class_collumn], axis = 1) #drop class data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y) #split
    tree_clf = DecisionTreeClassifier(random_state=0,max_depth=3) #select classifier
    tree_clf.fit(X_train,y_train) #fit classifier
    y_predicted = tree_clf.predict(X_test) #predict class values for test data
    accuracy = accuracy_score(y_test,y_predicted) #measure accuracy
    print("Accuracy:", round(accuracy,2))
    return accuracy
