from ScalingReal import dfArtists,dfGenres,dfPopularitySong,dfTName,df,dfEnergy,dfLoudness
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud






dfPopularityWithName = pd.concat([dfPopularitySong,dfTName],axis=1)
#-----------------------------------1
#Top 10 Artists with the Most Songs

ArtistsSong = dfArtists.value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=ArtistsSong.values, y=ArtistsSong.index, palette='viridis')
plt.xlabel('Song Count')
plt.ylabel('Artists')
plt.title('Top 10 Artists with the Most Songs')
plt.show()






#-----------------------------------2
#Genres Word Cloud
dfGenres_cleaned = dfGenres.str.replace('-', '_')
GenresFre = dfGenres.value_counts()
text =" ".join(dfGenres_cleaned)
wordcloud = WordCloud(width=800, height=400, background_color='white',colormap="tab10").generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title('Genres Word Cloud')
plt.show()








#----------------------------------3
#Variance Importance for all columns

df=df.drop(["x_1","x_2","x_3","x_4","x_5","x_6","x_7","artists_0","artists_1","artists_2","artists_3","artists_4","artists_5","artists_6","artists_7","artists_8","artists_9","artists_10","artists_11","artists_12","artists_13","artists_14","Energy-Loudness"],axis=1)
df = pd.concat([df,dfLoudness],axis=1)
df = pd.concat([df,dfEnergy],axis=1)
variances = df.var()
sorted_variances = variances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_variances, y=sorted_variances.index, palette="viridis")
plt.xlabel('Variance')
plt.title('Variance Importance for all columns')
plt.show()









#--------------------------------4
#Feature Distributions


num_columns = len(df.columns)
num_rows = 3
num_cols = num_columns // num_rows + (num_columns % num_rows > 0)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 10))

for i, column in enumerate(df.columns):
    ax = axs[i // num_cols, i % num_cols]
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'{column} Distribution')

plt.tight_layout()
plt.show()

