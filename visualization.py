# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
import string
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
stop=set(stopwords.words('english'))
nltk.download('stopwords')
stopwords = set(STOPWORDS)
plt.style.use('ggplot')

"""# Helper Method"""

# Word Cloud graph and save it to png
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()
    fig.savefig('Word Cloud.png', dpi=1000)

"""# Data Load"""

data = pd.read_csv('result.csv', index_col=0)

data.head()

"""# Text Visualization"""

corpus=[]
new= data['clean_tweet'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

a = show_wordcloud(corpus)

# we select top 20 most frequency words
top_N = 20

a = data['clean_tweet'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(a)
word_dist = nltk.FreqDist(words)
print (word_dist)

rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

rslt=rslt.sort_values('Frequency', ascending=True)

fig = plt.figure(figsize=(12, 8))
plt.barh(rslt['Word'], rslt['Frequency'])
fig.savefig('Word Frequency.png', dpi=1000)

"""# Sentiment analysis"""

data['Datetime'] = [data['Datetime'][i][:10] for i in range(len(data))]

data['Datetime'] =  pd.to_datetime(data['Datetime'], format='%Y-%m-%d')

temp =pd.DataFrame()
temp['datetime'] = data['Datetime']
temp['sentiment'] = data['result']

final = temp['sentiment'].groupby(temp['datetime']).mean()

fig = plt.figure(figsize=(12, 8))
final.plot()
plt.ylim(0, 1)
fig.savefig('final result.png', dpi=1000)