import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats
from scipy.stats import anderson
from scipy.stats import shapiro
from Cleaning import result
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.decomposition import PCA

df = result.copy()

columnNameMinMax = ["tempo","popularity","duration_ms"]

#Do Min-Max Scaling
def ScalingMinMax(df,columnNameMinMax):    
    min_max_scaler = preprocessing.MinMaxScaler()
    for name in columnNameMinMax:
         array = np.array(df[name])
         reshpaedArray = array.reshape(-1, 1)
         df[name] = min_max_scaler.fit_transform(reshpaedArray)
    return df

#Do Standart Scaling
def ScalingStandart(df,columnsNameStandart):
    scaler = StandardScaler()
    for name in columnsNameStandart:
         array = np.array(df[name])
         reshpaedArray = array.reshape(-1, 1)
         df[name] = scaler.fit_transform(reshpaedArray)
    return df

columnsNameStandart = ["danceability","loudness","valence"]
ScalingMinMax(df, columnNameMinMax)
ScalingStandart(df, columnsNameStandart)




#Information about dataset.
groups = df["track_genre"].value_counts()
groups = sorted(groups.keys())

#Drop unnecesary columns.
df.drop("Unnamed: 0",axis=1,inplace=True)

#Explict column transform to binary.
df["explicit"] = df["explicit"].astype(str)
df['explicit'] = df['explicit'].map({'True': 1, 'False': 0})

ghost = df.copy()
#Df have many duplicates drop them but one of them must stay.
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

#Get these specific dataframes use it later.
dfGenres = df["track_genre"]
dfTrack = df["track_id"]
dfTName = df["track_name"]
dfTurkish = df[df["track_genre"]=="turkish"]
dfArtists = df["artists"]

#Drop non-numeric column.
df = df.drop(["track_id","album_name","track_name","artists"],axis=1)




#One Hat Encoding.
#It is very large for dataset.
"""
encoding = pd.get_dummies(df["track_genre"])
encoding = encoding.applymap(lambda x: 1 if x == True else (0 if x == False else x))
df = pd.concat([df,encoding],axis=1)
"""

#Use Binary Encoder to transform genre to binary.
encoder = ce.BinaryEncoder(cols=["track_genre"])
df = encoder.fit_transform(df)
df = df.rename(columns={'track_genre_0': 'x_7','track_genre_1': 'x_6','track_genre_2': 'x_5','track_genre_3': 'x_4','track_genre_4': 'x_3','track_genre_5': 'x_2','track_genre_6': 'x_1'})


#Finding best correlation between the features.
def corrResult(min_corr, max_corr, data):
    for i in range(0, len(data.columns)):
        for j in range(0, len(data.columns)):
            if i != j:
                corr_1 = np.abs(data[data.columns[i]].corr(data[data.columns[j]],method = "spearman"))
                if corr_1 < min_corr:
                    continue
                elif corr_1 > max_corr:
                    print(data.columns[i], "and", data.columns[j],"high correlation with each other.")
                    

corrResult(min_corr = 0.3, max_corr = 0.75, data = df)



# Correlation matrix visualization.
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Corelasyon Matrisi')
plt.show()

# Covarince matrix visualization.
covariance_matrix = df.cov()
plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Kovaryans Matrisi')
plt.show()



#-------------PCA Algorithm.
dfEnergy = df["energy"]
dfLoudness = df["loudness"]
PCAdf = df[["energy","loudness"]]
X = PCAdf.values
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
df["Energy-Loudness"] = X_pca
df=df.drop(["energy","loudness"],axis=1)



df = pd.concat([dfArtists,df],axis=1)
dfPopularitySong = df["popularity"]


#Use Binary Encoder Method for artists column
encoder = ce.BinaryEncoder(cols=["artists"])
df = encoder.fit_transform(df)


dfLast = df 