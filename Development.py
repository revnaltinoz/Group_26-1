from ScalingReal import dfLast,dfTName
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from ScalingReal import dfTName,dfAlbum,dfArtists,dfTurkish

dfTurkish = dfTurkish


dfSelection = pd.concat([dfArtists,dfAlbum,dfTName],axis=1)
dfSelection = dfSelection.reset_index()
dfSelection = dfSelection.drop("index",axis=1)
dfLast = dfLast.reset_index()
dfLast = dfLast.drop("index",axis=1)
dfLabels = dfTName.reset_index()
# Assuming dfFeatures and dfLabels are defined properly
dfLast = dfLast.drop("key",axis=1)
dfFeatures = dfLast
dfLabels = dfTName
dfALL = pd.concat([dfFeatures, dfLabels], axis=1)

def getSong(df):
    """
    Artists=input("Artist Name : ")
    Album=input("Album Name : ")
    Track=input("Track Name : ")
    """
    Artists="Stabil;Ati242"
    Album="OKSMRN1"
    Track="MATHILDA"
    try:
        index=df[(df["artists"] == Artists) & (df["album_name"] == Album) & (df["track_name"] == Track)].index
        return index[0]
    except IndexError as e:
        return e
    
index=getSong(dfSelection)

inputsong = np.array(dfFeatures.loc[index])
print(dfFeatures.shape)
inputsong = inputsong.reshape(-1, len(dfFeatures.columns))  
inputsong = pd.DataFrame(inputsong)

#-----------------------KMeans Algorithm

kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(dfFeatures)

def find_closest_songs(userfeatures, dfFeatures, dfLabels, top_n=5):
    cluster_input = kmeans.predict(userfeatures)
    cluster_songs_indices = (kmeans.labels_ == cluster_input[0])
    
    cluster_songs_indices = cluster_songs_indices & ~(dfLabels.index == inputsong.index[0] if len(inputsong.index) > 0 else -1)
    cluster_songs = dfLabels[cluster_songs_indices]
    
    distances = euclidean_distances(userfeatures, dfFeatures[cluster_songs_indices].values)
    closest_song_indices = distances.argsort()[0][:top_n]
    
    closest_songs = cluster_songs.iloc[closest_song_indices]
    
    return closest_songs.index

closestsongsIndex = find_closest_songs(inputsong, dfFeatures, dfLabels)
ClosestSong = pd.DataFrame()

for i in closestsongsIndex:
    temp = dfSelection[dfSelection.index==i]
    ClosestSong = pd.concat([temp,ClosestSong])
    
    

#----------------------------------Editing Will be continue...


#--------------------------KNN Algorithm.
"""
k = 5
knn = NearestNeighbors(n_neighbors=k)
knn.fit(dfFeatures)
input_song_features = dfFeatures.loc[112171].values.reshape(1, -1)
distances, indices = knn.kneighbors(input_song_features)
closest_songs = dfLabels.iloc[indices[0]]
closest_songs_top_5 = closest_songs.head(5)

#----------------------------------Editing Will be continue...



#--------------------------Naive Bayes Algorithm.


gnb_model = GaussianNB()
gnb_model.fit(dfFeatures, dfLabels)
input_song_features = dfFeatures.loc[24202].values.reshape(1, -1)
input_song_probs = gnb_model.predict_proba(input_song_features)
all_songs_probs = gnb_model.predict_proba(dfFeatures)
similarities = cosine_similarity(input_song_probs, all_songs_probs)
similarities = normalize(similarities)
similar_indices = similarities.flatten().argsort()[::-1]
closest_song_indices = similar_indices[1:6]
closest_songs = dfLabels.iloc[closest_song_indices]
print(closest_songs)

#----------------------------------Editing Will be continue...


#--------------------------Birch Algorithm.


brc = Birch(n_clusters=50)  
brc.fit(dfFeatures)
input_song_cluster = brc.predict(dfFeatures.loc[24202].values.reshape(1, -1))
songs_in_cluster = dfLabels[brc.labels_ == input_song_cluster[0]]
input_song_features = dfFeatures.loc[24202].values.reshape(1, -1)
similarities = cosine_similarity(input_song_features, dfFeatures[brc.labels_ == input_song_cluster[0]])
similarities = normalize(similarities)
similar_indices = similarities.flatten().argsort()[::-1]
closest_song_indices = similar_indices[1:6]
closest_songs = dfLabels[songs_in_cluster.index[closest_song_indices]]
print("Top 5 closest songs within the cluster:")
print(closest_songs)

#----------------------------------Editing Will be continue...


#
#
# ------- Also we will use the Algorithms : Agglomerative
#
#




# Assuming dfFeatures contains numerical features for each song

n_clusters = 100  
mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
mbk.fit(dfFeatures)

k = 5
knn = NearestNeighbors(n_neighbors=k)

input_song_features = dfFeatures.loc[24202].values.reshape(1, -1)

input_song_cluster = mbk.predict(input_song_features)
cluster_samples = dfFeatures[mbk.labels_ == input_song_cluster[0]]

if len(cluster_samples) >= k:
    knn.fit(cluster_samples)
    distances, indices = knn.kneighbors(input_song_features)
    closest_songs = dfLabels.iloc[cluster_samples.iloc[indices[0]].index]
    closest_songs_top_5 = closest_songs.head(5)
else:
    print("Not enough samples in the cluster to find neighbors.")


#----------------------------------Editing Will be continue...


#We are currently developing the model and editing the codes we have written so far, which we will gradually edit. 
# We will also add different algorithms as extra.
"""