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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import category_encoders as ce
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


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

columnsNameStandart = ["danceability","loudness","valence","instrumentalness","key"]
ScalingMinMax(df, columnNameMinMax)
ScalingStandart(df, columnsNameStandart)




#Information about dataset.
groups = df["track_genre"].value_counts()
groups = sorted(groups.keys())

#Drop unnecesary columns.
df.drop("explicit",axis=1,inplace=True)
df.drop("Unnamed: 0",axis=1,inplace=True)

#Explict column transform to binary.
#df["explicit"] = df["explicit"].astype(str)
#df['explicit'] = df['explicit'].map({'True': 1, 'False': 0})

ghost = df.copy()
#Df have many duplicates drop them but one of them must stay.
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
# Error 1 = There is one music but this music has one more album in same time so it is bad for data.
df['track_name'] = df['track_name'].str.lower()

df.drop_duplicates(subset=['artists', 'track_name'], keep='first', inplace=True)
df = df.reset_index()
dfGenres = df["track_genre"]
dfTrack = df["track_id"]
dfTName = df["track_name"]
dfTurkish = df[df["track_genre"]=="turkish"]
dfArtists = df["artists"]
dfAlbum = df["album_name"]



    


#Get these specific dataframes use it later.


#Drop non-numeric column.
df = df.drop(["track_id","album_name","track_name","artists"],axis=1)




#One Hat Encoding.
#It is very large for dataset.
"""
encoding = pd.get_dummies(df["track_genre"])
encoding = encoding.applymap(lambda x: 1 if x == True else (0 if x == False else x))
df = pd.concat([df,encoding],axis=1)
"""


"""
#Use Binary Encoder to transform genre to binary.
encoder = ce.BinaryEncoder(cols=["track_genre"])
df = encoder.fit_transform(df)
df = df.rename(columns={'track_genre_0': 'x_7','track_genre_1': 'x_6','track_genre_2': 'x_5','track_genre_3': 'x_4','track_genre_4': 'x_3','track_genre_5': 'x_2','track_genre_6': 'x_1'})
df = df.drop("index",axis=1)
"""


df['track_genre'] = df['track_genre'].apply(lambda x: hash(''.join(x)))

scaler = MinMaxScaler(feature_range=(-100, 100))
df['track_genre'] = scaler.fit_transform(df[['track_genre']])





#Finding best correlation between the features.
def corrResult(min_corr, max_corr, data):
    for i in range(0, len(data.columns)):
        for j in range(0, len(data.columns)):
            if i != j:
                corr_1 = np.abs(data[data.columns[i]].corr(data[data.columns[j]],method = "spearman"))
                if corr_1 < min_corr:
                    continue
                elif corr_1 > max_corr:
                    print(data.columns[i], "and", data.columns[j],"High correlated each other.")
                    

corrResult(min_corr = 0.3, max_corr = 0.6, data = df)



# Correlation matrix visualization.
correlation_matrix = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlatin Matrix')
plt.show()

# Covarince matrix visualization.
covariance_matrix = df.cov()
plt.figure(figsize=(15, 9))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Covariance Matrix')
plt.show()



#-------------PCA Algorithm.
dfEnergy = df["energy"]
dfLoudness = df["loudness"]
PCAdf = df[["energy","loudness"]]
X = PCAdf.values
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
df["Eng-Loudn"] = X_pca
df=df.drop(["energy","loudness"],axis=1)


#Deneme artist
"""
PCAdf = df[["x_7","x_6","x_5","x_4","x_3","x_2","x_1"]]
X = PCAdf.values
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
df["deneme genre "] = X_pca
df=df.drop(["x_7","x_6","x_5","x_4","x_3","x_2","x_1"],axis=1)
"""





#-------------------------Svd for genre

"""
# Öncelikle belirli özellikleri seçin ve sütun adlarını değiştirin
SVD_df = df[["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"]]
X = SVD_df.values
# TruncatedSVD uygulama
n_components = 1  # Elde etmek istediğiniz bileşen sayısı
svd = TruncatedSVD(n_components=n_components)
X_svd = svd.fit_transform(X)

# DataFrame'e dönüştürme
df["deneme genre"] = X_svd
df = df.drop(["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"], axis=1)

"""




"""
PCAdf = df[["x_7","x_6","x_5","x_4","x_3","x_2","x_1"]]
X = PCAdf.values
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
df["deneme artists "] = X_pca
df=df.drop(["x_7","x_6","x_5","x_4","x_3","x_2","x_1"],axis=1)
"""










def find_least_correlated_column(df):
    correlation_matrix = df.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    absolute_correlation = upper_triangle.abs()
    least_correlated_column = absolute_correlation.min().idxmin()
    
    return least_correlated_column


MinCorrelated = find_least_correlated_column(df)
#df = df.drop(MinCorrelated,axis=1)



df = pd.concat([dfArtists,df],axis=1)





dfPopularitySong = df["popularity"]

"""
#Use Binary Encoder Method for artists column
encoder = ce.BinaryEncoder(cols=["artists"])
df = encoder.fit_transform(df)
"""




#----------------------SVD for artists
"""
# Öncelikle belirli özellikleri seçin ve sütun adlarını değiştirin
SVD_df = df[["artists_0", "artists_1", "artists_2", "artists_3", "artists_4", "artists_5", "artists_6", "artists_7", "artists_8", "artists_9", "artists_10", "artists_11", "artists_12", "artists_13", "artists_14"]]
X = SVD_df.values

# TruncatedSVD uygulama
n_components = 1  # Elde etmek istediğiniz bileşen sayısı
svd = TruncatedSVD(n_components=n_components)
X_svd = svd.fit_transform(X)

# DataFrame'e dönüştürme
df["denemeartists"] = X_svd
df = df.drop(["artists_0", "artists_1", "artists_2", "artists_3", "artists_4", "artists_5", "artists_6", "artists_7", "artists_8", "artists_9", "artists_10", "artists_11", "artists_12", "artists_13", "artists_14"], axis=1)
"""

def merge_names(names):
    if ';' in names:
        # Virgülle ayrılmış isimleri işle
        name_list = names.split(';')
        merged_names = []
        for name in name_list:
            # Boşlukları kaldırarak isim ve soyisimleri birleştir
            merged_names.append(''.join(name.split()))
        return ''.join(merged_names)
    else:
        # Tek bir isim varsa boşlukları kaldırarak birleştir
        return ''.join(names.split())

df['artists'] = df['artists'].apply(merge_names)


# Her bir sanatçı adını benzersiz sayısal değerle değiştirme
df['artists'] = df['artists'].apply(lambda x: hash(''.join(x)))

scaler = MinMaxScaler(feature_range=(-100, 100))
df['artists'] = scaler.fit_transform(df[['artists']])



dfLast = df 
