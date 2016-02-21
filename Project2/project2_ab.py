# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:11:50 2016

@author: fengjun
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.ticker import FormatStrFormatter
import nltk.stem


english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        

cat=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = f20(subset='train',categories=cat, shuffle = True, random_state = 42)


x=np.arange(0,9,1)
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(train.target,x, facecolor='red', edgecolor='gray')
ax.set_xticks(bins)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Targets',x=1)
ax.set_ylabel('Numbers')
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the raw counts
    ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.0001f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')
plt.title('Number of Documents per Target')
plt.subplots_adjust(bottom=0.15)

train = f20(subset='train',shuffle = True, random_state = 42)

stopwords = text.ENGLISH_STOP_WORDS
vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words=stopwords, decode_error='ignore')
vector = vectorizer.fit_transform(train.data)
tfidf=vector.toarray()
print 'number of terms: '+str(len(tfidf[0]))
cat={0:'alt.atheism',1:'alt.atheism',2:'comp.graphics',3:'comp.os.ms-windows.misc',4:'comp.sys.ibm.pc.hardware',5:'comp.windows.x',6:'misc.forsale',7:'rec.autos',8:'rec.motorcycles',9:'rec.sport.baseball',10:'rec.sport.hockey',11:'sci.crypt',12:'sci.electronics',13:'sci.med',14:'sci.space',15:'soc.religion.christian',16:'talk.politics.guns',17:'talk.politics.mideast',18:'talk.politics.misc',19:'talk.religion.misc'
}
number_of_document=[]
raw_data=[]
for i in range(0,20):
    tmp_data=f20(subset='train',categories=[cat[i]], shuffle = True, random_state = 42)
    number_of_document.append(len(tmp_data.data))
    raw_data.append(tmp_data.data)
intend_number=min(number_of_document)
for i in range(0,20):
    print len(raw_data[i])
    raw_data[i]=raw_data[i][0:intend_number]
    print len(raw_data[i])
new_data=[]
new_target=[]
for i in range(0,20):
    for j in range(0,intend_number):
        new_data.append(raw_data[i][j])
        new_target.append(i)