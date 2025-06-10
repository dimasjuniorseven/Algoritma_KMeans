import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
df = pd.read_csv('C:/Users/ASUS/DimasOktavianPrasetyo_22230016.csv')  # Pastikan path benar
tweets = df['Isi Tweet'].dropna().tolist()

# Preprocessing ringan + hapus stopword Indonesia
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

def preprocess(text):
    text = re.sub(r"http\S+", "", text)  # Hapus link
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Hapus simbol/karakter non-alfabet
    text = text.lower()
    tokens = text.split()
    return ' '.join([w for w in tokens if w not in stop_words])

# Preprocessing semua tweet
tweets_cleaned = [preprocess(t) for t in tweets]

# Vektorisasi TF-IDF
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(tweets_cleaned)

# Clustering dengan KMeans
k = 5
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Tampilkan hasil
for i in range(k):
    print(f"\nCluster {i + 1}:")
    indices = [j for j, label in enumerate(model.labels_) if label == i]
    for idx in indices[:5]:  # Tampilkan 5 tweet pertama dari cluster ini
        print(f"- {tweets[idx]}")
