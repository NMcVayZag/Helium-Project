import pandas as pd
import numpy as np
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
