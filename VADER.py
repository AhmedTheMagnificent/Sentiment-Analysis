import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
plt.style.use('ggplot')

df = pd.read_csv(r"C:\Users\ahmed\Downloads\archive (1)\Reviews.csv")
df = df.head(1000)
token = nltk.word_tokenize(df['Text'][50])
pos = nltk.pos_tag(token)
chunk = nltk.chunk.ne_chunk(pos)

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("I am so happy that i can kill someone"))