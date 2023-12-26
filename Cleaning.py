import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats
from scipy.stats import median_abs_deviation



df = pd.read_csv("dataset.csv")
nanvalues = df.isna().sum()
df = df.dropna()



df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
# Error 1 = There is one music but this music has one more album in same time so it is bad for data.
df['track_name'] = df['track_name'].str.lower()

df.drop_duplicates(subset=['artists', 'track_name'], keep='first', inplace=True)

dfTempo = df["tempo"]
ghost = df

def checkNormal(df,name):
    #plt.title(name,fontdict=16)
    sns.histplot(df[name], kde=True)
    return plt.show()

def Zscore(df,name):
    zScore = stats.zscore(df[name])
    threshold = 3
    outliers = df[name][abs(zScore)>threshold]
    return df.drop(df[df[name].isin(outliers)].index)

def IQR(df,name):
    Q1 = df[name].quantile(0.25)
    Q3 = df[name].quantile(0.75)  
    IQR = Q3 - Q1
    LowerFence = Q1 - 1.5 * IQR
    UpperFence = Q3 + 1.5 * IQR
    return df.drop(df[(df[name] < LowerFence) | (df[name] > UpperFence)].index)


def Robust(df,name):
    median = df[name].median()
    mad = median_abs_deviation(df[name])
    robust_z_scores = (df[name] - median) / mad
    threshold = 4
    outliers = df[name][abs(robust_z_scores) > threshold]    
    return df.drop(outliers.index)
    
    
    
    
    
    
    
    
    
# duration_ms için işlem
#-----------
#checkNormal(df, "duration_ms")
#df = Zscore(df, "duration_ms") #414
df = Robust(df, "duration_ms") #12043
#df = IQR(df, "duration_ms") #5616
#checkNormal(df, "duration_ms")






# popularity için işlem
#-----------
#checkNormal(df, "popularity")
#df = Zscore(df, "popularity") #0
df = Robust(df, "popularity") #475
#df = IQR(df, "popularity") #2
#checkNormal(df, "popularity")





# explicit için işlem
#-----------
#checkNormal(df, "explicit")
#df = Zscore(df, "explicit") #0
#df = Robust(df, "explicit") #9747
#df = IQR(df, "explicit") #dont work
#checkNormal(df, "explicit")

#Explicit i silebiliriz cunku explicit sarkının explicit lyrics i var mı yok mu onu gosteriyor
#bu da muzik onerisi icin onemli olmamalı bence







# danceability için işlem
#-----------
#checkNormal(df, "danceability")
#df = Zscore(df, "danceability") #162
df = Robust(df, "danceability") #4281
#df = IQR(df, "danceability")
#checkNormal(df, "danceability")






#hic bir cleaning azaltmıyor bunu cuk dağılmış heralde

# key için işlem
#-----------
#checkNormal(df, "key")
#df = Zscore(df, "key")
#df = Robust(df, "key")
#df = IQR(df, "key")
#checkNormal(df, "key")








# mode için işlem
#-----------
#checkNormal(df, "mode")
#df = Zscore(df, "mode")
#df = Robust(df, "mode")
#df = IQR(df, "mode")
#checkNormal(df, "mode")





# speechiness için işlem
#-----------
#checkNormal(df, "speechiness")
#df = Zscore(df, "speechiness") #2000
#df = Robust(df, "speechiness") # 20000
df = IQR(df, "speechiness") #10000
#checkNormal(df, "speechiness")






# acousticness için işlem
#-----------
#checkNormal(df, "acousticness")
#df = Zscore(df, "acousticness")
#df = Robust(df, "acousticness") #16000
#df = IQR(df, "acousticness") #0
#checkNormal(df, "acousticness")








# instrumentalness için işlem
#-----------
#checkNormal(df, "instrumentalness")
df = Zscore(df, "instrumentalness") #4000
#df = Robust(df, "instrumentalness")
#df = IQR(df, "instrumentalness")
#checkNormal(df, "instrumentalness")









# liveness için işlem
#-----------
#checkNormal(df, "liveness")
#df = Zscore(df, "liveness")
df = Robust(df, "liveness")
#df = IQR(df, "liveness")
#checkNormal(df, "liveness")









# valence için işlem
#-----------
#checkNormal(df, "valence")
df = Zscore(df, "valence")
#df = Robust(df, "valence")
#df = IQR(df, "valence")
#checkNormal(df, "valence")








# tempo için işlem
#-----------
#checkNormal(df, "tempo")
df = Zscore(df, "tempo")
#df = Robust(df, "tempo")
#df = IQR(df, "tempo")
#checkNormal(df, "tempo")









# time_signature için işlem
#-----------
#checkNormal(df, "time_signature")
#df = Zscore(df, "time_signature")
#df = Robust(df, "time_signature")
#df = IQR(df, "time_signature")
#checkNormal(df, "time_signature")






    
    
    
    
"""    
checkNormal(df, "duration_ms")
df = Robust(df, "popularity")
#df = Robust(df, "duration_ms")
checkNormal(df, "duration_ms")





df=Zscore(df,"tempo")
df=Zscore(df,"danceability")
df=IQR(df,"energy")
df=Zscore(df,"loudness")
df=Zscore(df,"speechiness")
df=Zscore(df,"valence")
df=IQR(df,"liveness")
#df=IQR(df,"popularity")
"""
print(df['track_id'].nunique(), df.shape)


result = df

