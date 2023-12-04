import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats


df = pd.read_csv("dataset.csv")
nanvalues = df.isna().sum()
df = df.dropna()

dfTempo = df["tempo"]
ghost = df

def checkNormal(df,name):
    sns.histplot(df[name], kde=True)
    return plt.show()

def Zscore(df,name):
    zScore = stats.zscore(df[name])
    threshold = 4
    outliers = df[name][abs(zScore)>threshold]
    return df.drop(df[df[name].isin(outliers)].index)

def IQR(df,name):
    Q1 = df[name].quantile(0.25)
    Q3 = df[name].quantile(0.75)  
    IQR = Q3 - Q1
    LowerFence = Q1 - 1.5 * IQR
    UpperFence = Q3 + 1.5 * IQR
    return df.drop(df[(df[name] < LowerFence) | (df[name] > UpperFence)].index)

df=Zscore(df,"tempo")
df=Zscore(df,"danceability")
df=IQR(df,"energy")
df=Zscore(df,"loudness")
df=Zscore(df,"speechiness")
df=Zscore(df,"valence")
df=IQR(df,"liveness")
df=IQR(df,"popularity")

print(df['track_id'].nunique(), df.shape)


result = df

