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

