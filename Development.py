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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from ScalingReal import dfTName,dfAlbum,dfArtists,dfTurkish
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler




dfTurkish = dfTurkish


dfSelection = pd.concat([dfArtists,dfAlbum,dfTName],axis=1)
dfSelection = dfSelection.reset_index()
dfSelection = dfSelection.drop("index",axis=1)
dfLast = dfLast.reset_index()
dfLast = dfLast.drop("index",axis=1)
dfLast = dfLast.drop("level_0",axis=1)

dfLabels = dfTName.reset_index()

# Assuming dfFeatures and dfLabels are defined properly
dfFeatures = dfLast
dfLabels = dfTName
dfALL = pd.concat([dfFeatures, dfLabels], axis=1)

def getSong(df):
    """
    Artists=input("Artist Name : ")
    Album=input("Album Name : ")
    Track=input("Track Name : ")
    """
    Artists="Hidra"
    Album="Sniper"
    Track="sniper"
    try:
        index=df[(df["artists"] == Artists) & (df["album_name"] == Album) & (df["track_name"] == Track)].index
        return index[0]
    except IndexError as e:
        return e
    
inputIndex=getSong(dfSelection)
selectedIndex = dfFeatures.loc[inputIndex]
inputsong = pd.DataFrame(selectedIndex).T

#-----------------------KMeans Algorithm
"""

def KmeansPrediction(userfeatures,dfFeatures,dfLabels,number):
    kmeans = KMeans(n_clusters=3, random_state=10, n_init=3)
    kmeans.fit(dfFeatures)
    clusterofInput = kmeans.predict(inputsong)
    cluster_songs_indices = (kmeans.labels_ == clusterofInput[0])
    cluster_songs_indices = cluster_songs_indices & ~(dfFeatures.index == inputsong.index[0])
    cluster_songs = dfLabels[cluster_songs_indices]
    distances = euclidean_distances(inputsong, dfFeatures[cluster_songs_indices].values)
    closest_song_indices = distances.argsort()[0]
    closest_songs = cluster_songs.iloc[closest_song_indices[:number]]
    return closest_songs.index, kmeans.labels_

indexesKmeans,predicted_labels= KmeansPrediction(inputsong, dfFeatures, dfLabels, 5)
ClosestSongs = pd.DataFrame()
for i in indexesKmeans:
    temp = dfSelection[dfSelection.index == i]
    ClosestSongs = pd.concat([temp, ClosestSongs]) 


silhouette = silhouette_score(dfFeatures, predicted_labels)
print("Silhouette Score:", silhouette)

"""
#----------------------------------Editing Will be continue...


#--------------------------KNN Algorithm.

"""
def KnnPrediction(dfFeatures,number,inputsong,dfLabels):
    knn = NearestNeighbors(n_neighbors=number)
    knn.fit(dfFeatures)
    distances, indices = knn.kneighbors(inputsong)
    indexes = indices[0]
    indexes = indexes[indexes != inputsong.index[0]]
    return indexes
    
indexesKnn= KnnPrediction(dfFeatures, 6, inputsong, dfLabels)
ClosestSongs = pd.DataFrame()
for i in indexesKnn:
    temp = dfSelection[dfSelection.index == i]
    ClosestSongs = pd.concat([temp, ClosestSongs]) 
"""


#----------------------------------Editing Will be continue...


#----------------------------------MiniBatch
"""
def MiniBatchPrediction(inputsong,dfFeatures,dfLabels,numberofclusters,numberK):
    mbk = MiniBatchKMeans(n_clusters=numberofclusters, random_state=42)
    mbk.fit(dfFeatures)
    knn = NearestNeighbors(n_neighbors=numberK)
    input_song_cluster = mbk.predict(inputsong)
    cluster_samples = dfFeatures[mbk.labels_ == input_song_cluster[0]]
    if len(cluster_samples) >= numberK:
        knn.fit(cluster_samples)
        distances, indices = knn.kneighbors(inputsong)
    else:
        print("Not enough samples in the cluster to find neighbors.")
    indexes = indices[0]
    indexes = indexes[indexes != inputsong.index[0]]
    return indexes
     
indexesMiniBatch = MiniBatchPrediction(inputsong, dfFeatures,dfLabels,50,6)
ClosestSongs = pd.DataFrame()

for i in indexesMiniBatch:
    temp = dfSelection[dfSelection.index == i]
    ClosestSongs = pd.concat([temp, ClosestSongs]) 
"""  

#Elbow Methodu kullanılması gerekiyor numberofcluster sayısını dogru bir sekilde bulmak icin
#Bu method ile kumeleme performansının nasıl degistigi gozlenebilir.
#Revna?
    
#--------------------------------Editing will be continue...


















#--------------------------Birch Algorithm.
"""
def BirchPrediction(inputsong, dfFeatures, dfLabels):
    brc = Birch(n_clusters=2,branching_factor=50,threshold=.5)  
    brc.fit(dfFeatures)
    input_song_cluster = brc.predict(inputsong)
    cluster_features = dfFeatures[brc.labels_ == input_song_cluster[0]]
    
    if len(cluster_features) > 0:
        similarities = euclidean_distances(inputsong, cluster_features)
        similarities = normalize(similarities)
        similar_indices = similarities.flatten().argsort()[::-1]
        closest_song_indices = similar_indices[1:6]  # Excluding the input song itself
        
        return closest_song_indices
    else:
        return None

indexesBirchPrediction = BirchPrediction(inputsong, dfFeatures, dfLabels)
ClosestSongs = pd.DataFrame()

for i in indexesBirchPrediction:
    temp = dfSelection[dfSelection.index == i]
    ClosestSongs = pd.concat([temp, ClosestSongs]) 

"""

"""              
silhouette_scores = []

for n_clusters in range(2,5):  # Trying different values of k
    brc = Birch(n_clusters=n_clusters)
    cluster_labels = brc.fit_predict(dfFeatures)
    silhouette_avg = silhouette_score(dfFeatures, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting the Silhouette scores
plt.plot(range(2, 5), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.show()
"""


#----------------------------------Editing Will be continue...
















#------------------------------Agglomerative


"""

def AgglomerativePrediction(inputsong, dfFeatures, numberclosest):
    cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='average')
    cluster.fit(dfFeatures)
    cluster_labels = cluster.labels_
    input_song_cluster = cluster_labels[inputsong.index[0]]

    similar_songs_indexes = [index for index, label in enumerate(cluster_labels) if label == input_song_cluster and index != inputsong.index[0]]

    similarities = [(index, dfFeatures.iloc[inputsong.index[0]].corr(dfFeatures.iloc[index])) for index in similar_songs_indexes]
    similarities.sort(key=lambda x: x[1], reverse=True)

    closest_songs_indexes = [index for index, _ in similarities[:numberclosest]]

    return closest_songs_indexes

closest_songs_agglomerative = AgglomerativePrediction(inputsong, dfFeatures, 5)
ClosestSongsAgglomerative = dfSelection.iloc[closest_songs_agglomerative]
"""

#----------------------------Editing will be continue













#--------------------------Naive Bayes Algorithm.
"""

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

"""



"""

def DBSCAN_Prediction(inputsong, dfFeatures, eps_value, min_samples_value):
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan.fit(dfFeatures)
    cluster_labels = dbscan.fit_predict(dfFeatures)
    input_song_cluster = dbscan.fit_predict(inputsong.values.reshape(1, -1))
    cluster_indices = np.where(cluster_labels == input_song_cluster)[0]
    if len(cluster_indices) > 0:
        cluster_features = dfFeatures.iloc[cluster_indices]
        similarities = euclidean_distances(inputsong.values.reshape(1, -1), cluster_features)
        similarities = normalize(similarities)
        similar_indices = similarities.flatten().argsort()[::-1]
        closest_song_indices = similar_indices[1:6]
        return closest_song_indices

    return None

eps_value = 0.5  
min_samples_value = 5 
indexesDBSCANPrediction = DBSCAN_Prediction(inputsong, dfFeatures, eps_value, min_samples_value)
ClosestSongs = pd.DataFrame()

if indexesDBSCANPrediction is not None:
    for i in indexesDBSCANPrediction:
        temp = dfSelection[dfSelection.index == i]
        ClosestSongs = pd.concat([temp, ClosestSongs]) 


"""



"""


def calculate_silhouette_scores(kmeans, data):
    silhouette_scores = []
    n_init_range = range(2, 11)  # Define the range for n_init values
    
    for n_init_value in n_init_range:
        kmeans.set_params(n_init=n_init_value)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette = silhouette_score(data, labels)
        silhouette_scores.append(silhouette)
    
    return silhouette_scores

kmeans = KMeans(n_clusters=3, random_state=42)
silhouette_scores = calculate_silhouette_scores(kmeans, dfFeatures)

# Plotting silhouette scores against n_init values
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different n_init values')
plt.xlabel('n_init')
plt.ylabel('Silhouette Score')
plt.show()

# Calculate and print the silhouette score for the chosen KMeans model
kmeans.fit(dfFeatures)
kmeans_score = silhouette_score(dfFeatures, kmeans.labels_)
print(f"KMeans Silhouette Score: {kmeans_score}")

"""

def calculate_silhouette_score(model, features):
    labels = model.fit_predict(features)
    silhouette_avg = silhouette_score(features, labels)
    return silhouette_avg



# KMeans
kmeans = KMeans(n_clusters=3, random_state=10, n_init=3)
kmeans_score = calculate_silhouette_score(kmeans, dfFeatures)
print(f"KMeans Silhouette Score REVNAAAAAAAAAAAAAAAA : {kmeans_score}")

# MiniBatchKMeans
#mini_batch_kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=3)
#mini_batch_kmeans_score = calculate_silhouette_score(mini_batch_kmeans, dfFeatures)
#print(f"MiniBatchKMeans Silhouette Score: {mini_batch_kmeans_score}")

"""
def calculate_silhouette_scores(param_name, param_range, fixed_params, data):
    silhouette_scores = []
    
    for param_value in param_range:
        params = fixed_params.copy()
        params[param_name] = param_value
        
        kmeans = MiniBatchKMeans(**params)
        kmeans.fit(data)
        labels = kmeans.labels_
        
        silhouette = silhouette_score(data, labels)
        silhouette_scores.append(silhouette)
    
    return silhouette_scores

# Define the range of batch_size values to test
batch_size_range = [1500,2000,2500]

# Fixed parameters for MiniBatchKMeans
fixed_params = {
    'n_clusters': 3,  # Set other parameters to constant values
    'max_iter': 100,
    'init': 'k-means++'
}

# Calculate silhouette scores for different batch_size values
silhouette_scores = calculate_silhouette_scores('batch_size', batch_size_range, fixed_params, dfFeatures)

# Plotting silhouette scores against batch_size values
plt.plot(batch_size_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different batch_size values')
plt.xlabel('batch_size')
plt.ylabel('Silhouette Score')
plt.show()
"""


# Birch
#birch = Birch(n_clusters=3)
#birch_score = calculate_silhouette_score(birch, dfFeatures)
#print(f"Birch Silhouette Score: {birch_score}")

# AgglomerativeClustering
#agglomerative = AgglomerativeClustering(n_clusters=20)
#agglomerative_score = calculate_silhouette_score(agglomerative, dfFeatures_scaled)
#print(f"AgglomerativeClustering Silhouette Score: {agglomerative_score}")

# Naive Bayes (Assuming dfFeatures and dfLabels are features and labels)
#naive_bayes = GaussianNB()
#naive_bayes.fit(dfFeatures_scaled, dfLabels)
#naive_bayes_score = silhouette_score(dfFeatures_scaled, naive_bayes.predict(dfFeatures_scaled))
#print(f"Naive Bayes Silhouette Score: {naive_bayes_score}")


#------------------------------------------------------- Elbow methods

#For minibatch
"""

def find_optimal_clusters(features, max_clusters=10):
    inertias = []
    for i in range(1, max_clusters + 1):
        mbk = MiniBatchKMeans(n_clusters=i, random_state=42)
        mbk.fit(features)
        inertias.append(mbk.inertia_)

    return inertias

# Assuming dfFeatures is defined properly
dfFeatures_scaled = StandardScaler().fit_transform(dfFeatures)

# Find optimal number of clusters using the elbow method
max_clusters_to_try = 20
inertias = find_optimal_clusters(dfFeatures_scaled, max_clusters_to_try)

# Plot the elbow curve
plt.plot(range(1, max_clusters_to_try + 1), inertias, marker='o')
plt.title('Elbow Method for MiniBatchKMeans')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
"""

#------------------------------------------------for kmeans
"""
def find_optimal_clusters(features, max_clusters=10):
    inertias = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    return inertias


max_clusters_to_try = 20
inertias = find_optimal_clusters(dfFeatures, max_clusters_to_try)

# Plot the elbow curve
plt.plot(range(1, max_clusters_to_try + 1), inertias, marker='o')
plt.title('Elbow Method for KMeans')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()



from scipy.signal import find_peaks

def find_optimal_clusters(features, max_clusters=10):
    inertias = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Calculate first derivative
    derivative = np.gradient(inertias)

    # Find peaks in the derivative
    peaks, _ = find_peaks(derivative, distance=1)

    # The first peak indicates the elbow point
    if len(peaks) > 0:
        return peaks[0] + 1  # Adding 1 because of zero-based indexing (clusters start from 1)
    else:
        return None

max_clusters_to_try = 50
optimal_clusters = find_optimal_clusters(dfFeatures, max_clusters_to_try)
print("Optimal Number of Clusters (Elbow Point):", optimal_clusters)
"""

#-------------------------------------------------------davies bouldin ve calinski harabasz
"""
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming dfFeatures is defined properly
dfFeatures_scaled = StandardScaler().fit_transform(dfFeatures)

def KmeansPrediction(userfeatures, dfFeatures, dfLabels, number):
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for n_clusters in range(2, 21):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(dfFeatures)
        labels = kmeans.labels_

        davies_bouldin_score_value = davies_bouldin_score(dfFeatures_scaled, labels)
        calinski_harabasz_score_value = calinski_harabasz_score(dfFeatures_scaled, labels)

        davies_bouldin_scores.append(davies_bouldin_score_value)
        calinski_harabasz_scores.append(calinski_harabasz_score_value)

    # Plot Davies-Bouldin Index scores for different cluster numbers
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, 21), davies_bouldin_scores, marker='o')
    plt.title('Davies-Bouldin Index for KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')

    # Plot Calinski-Harabasz Index scores for different cluster numbers
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 21), calinski_harabasz_scores, marker='o', color='orange')
    plt.title('Calinski-Harabasz Index for KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Index')

    plt.tight_layout()
    plt.show()

    # Choose the number of clusters based on indices
    optimal_clusters_davies_bouldin = np.argmin(davies_bouldin_scores) + 2
    optimal_clusters_calinski_harabasz = np.argmax(calinski_harabasz_scores) + 2

    print(f'Optimal Number of Clusters (Davies-Bouldin): {optimal_clusters_davies_bouldin}') #bu printler calismiyor anlamadim
    print(f'Optimal Number of Clusters (Calinski-Harabasz): {optimal_clusters_calinski_harabasz}')

    # Choose the number of clusters with the lowest Davies-Bouldin Index
    kmeans = KMeans(n_clusters=optimal_clusters_davies_bouldin, random_state=42)
    kmeans.fit(dfFeatures)
    
    clusterofInput = kmeans.predict(inputsong)
    cluster_songs_indices = (kmeans.labels_ == clusterofInput[0])
    cluster_songs_indices = cluster_songs_indices & ~(dfFeatures.index == inputsong.index[0])
    cluster_songs = dfLabels[cluster_songs_indices]
    
    distances = euclidean_distances(inputsong, dfFeatures[cluster_songs_indices].values)
    closest_song_indices = distances.argsort()[0]
    closest_songs = cluster_songs.iloc[closest_song_indices[:number]]

    return closest_songs.index

# Example usage
indexesKmeans = KmeansPrediction(inputsong, dfFeatures, dfLabels, 5)
ClosestSongsKmeans = pd.DataFrame()

for i in indexesKmeans:
    temp = dfSelection[dfSelection.index == i]
    ClosestSongsKmeans = pd.concat([temp, ClosestSongsKmeans]) 

print(ClosestSongsKmeans)
"""

#----------------------------------------------silhouette visualization 
"""
#for kmeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt  # Ensure scikit-plot is installed

# Assuming dfFeatures is defined properly
dfFeatures_scaled = StandardScaler().fit_transform(dfFeatures)

def calculate_silhouette_score(model, data):
    labels = model.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg
"""
"""
# Function to plot silhouette scores for different numbers of clusters
def plot_silhouette_scores(data, max_clusters=20): #max cluster 20 dedim hala 10 cikiyo grafikte onu anlamadim 
    silhouette_scores = []
    clusters_range = range(2, max_clusters + 1)

    for n_clusters in clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_score_value = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_score_value)

    # Plot Silhouette Scores for different numbers of clusters
    plt.plot(clusters_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for KMeans Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Plot Silhouette Scores for different numbers of clusters
plot_silhouette_scores(dfFeatures_scaled, max_clusters=10)


"""

#----------------------------------Editing Will be continue...
"""

#We are currently developing the model and editing the codes we have written so far, which we will gradually edit. 
# We will also add different algorithms as extra.
"""
