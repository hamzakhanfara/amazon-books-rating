import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

nltk.download('vader_lexicon')
print('done importing')

data = pd.read_csv('dataset/Books_rating.csv')
print(data.head(5))

#Nettoyer les données en éliminant les utilisateurs qui ont fait un seul commentaire

user_counts = data['User_id'].value_counts()
data = data[data['User_id'].isin(user_counts[user_counts > 1].index)]

data.to_csv("cleaned_dataset/cleaned_data.csv",index=False)
print('done saving data')