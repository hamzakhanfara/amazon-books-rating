import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('cleaned_dataset/cleaned_data.csv')
print("#Créer une instance du SentimentIntensityAnalyzer de NLTK")

sia = SentimentIntensityAnalyzer()

print("#Calculer la valeur du score du commentaire normalisée")

data['normalized_score'] = data['review/text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

print("#Normaliser les scores dans l'intervalle [0, 1]")

scaler = MinMaxScaler()
data['normalized_score'] = scaler.fit_transform(data[['normalized_score']])
print("#Créer le score hybride")

data['hybrid_score'] = data['review/score'] * data['normalized_score']

print(data[['normalized_score','review/score', 'hybrid_score']].head())