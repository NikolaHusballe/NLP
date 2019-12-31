import pandas as pd
import os
from pathlib import Path
import re
from os.path import join
import glob
import csv
import io


'''
--------------- Reading twitter + news data --------------------------------------------------
'''

twt = pd.read_csv('climate_3_tweets.csv')

# drop tweets without urls
# twt_urls = twt.dropna(subset=['urls'])
# twt_urls.to_csv('twt_urls.csv')

# twt = pd.read_csv('twt_urls.csv')
news_data = pd.read_csv('news_data.csv')

#twt.head(5)

data = pd.read_csv('full_data.csv')
data = data[data['lang'] == 'en']


# data = pd.read_csv('twt_eng.csv')

del data['tweet_length']

# merged_urls.to_csv('full_data.csv')

# news = news.rename(columns={'0': 'ID', '1': 'wordID', '2': 'date', '3':'country','4':'sourceTitle', '5':'urls', '6':'textTitle',})
# news.to_csv('news_data.csv')


'''
--------------- cleaning the news data --------------------------------------------------
'''

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re



data['textTitle'] = data['textTitle'].astype(str)




# removing stopwords, replacing symbols w space, and deleting symbols
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;-]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+]')
STOPWORDS = set(stopwords.words('english'))
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
wordnet_lemmatizer = WordNetLemmatizer()

# editing stopwords list

STOPWORDS.remove('below')
STOPWORDS.remove('under')
STOPWORDS.remove('down')
STOPWORDS.remove('above')
STOPWORDS.remove('against')
STOPWORDS.remove('over')
STOPWORDS.remove('between')
STOPWORDS.add('climate')
STOPWORDS.add('change')
STOPWORDS.add('rt')
STOPWORDS.add('http')
STOPWORDS.add('u')
STOPWORDS.add('via')
STOPWORDS.add('go')
STOPWORDS.add('climatechange')
STOPWORDS.add('globalwarming')
STOPWORDS.add('global')
STOPWORDS.add('warming')
STOPWORDS.add('c')
STOPWORDS.add('could')


def clean_text(text, bigrams=False, lemma=False):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text_token_list = [word for word in text.split(' ') if word not in STOPWORDS]

    if lemma == True:
      text_token_list = [wordnet_lemmatizer.lemmatize(word) if '#' not in word else word 
      for word in text_token_list] # apply lemmatizer
    else:
      text_token_list = [word_rooter(word) if '#' not in word else word 
      for word in text_token_list] # apply word rooter
    if bigrams:
      text_token_list = text_token_list+[text_token_list[i]+'_'+text_token_list[i+1]
      for i in range(len(text_token_list)-1)]
    text = ' '.join(text_token_list)
    return text

# adding clean text to new variable
data['cl_textTitle'] = data['textTitle'].apply(clean_text, lemma=True, bigrams=True)

data.head(10)

#We create a new column with tokens
data['token_text'] = [
    [word for word in text.split()  if word not in STOPWORDS]
    for text in data['cl_textTitle']]

print(data['token_text'])



'''
--------------- Frequency NEWS ---------------------------------------------------------
'''

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# source frequency
# summary 
print(len(data['sourceTitle'].unique()))
# 611 unique sources

source = data.groupby('sourceTitle')

plt.figure(figsize=(15,10))
plt.title('10 most represented News sources', fontsize=16)
data['sourceTitle'].value_counts()[:10].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Article source")
plt.ylabel("Number of mentions")
plt.show()

# summary by country of origin

country = data.groupby('country')
print(len(data['country'].unique()))
# all countries represented in subset

plt.figure(figsize=(15,10))
plt.title('10 most represented countries', fontsize=16)
data['country'].value_counts()[:10].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Countries of sources")
plt.ylabel("Number mentions")
plt.show()

'''
--------------- BUILDING DICTIONARY + CORPUS ---------------------------------------
'''
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
from pprint import pprint
from collections import OrderedDict
import seaborn as sns
from gensim.models import CoherenceModel

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings

import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now



# only using unique headlines

news_unique = data.drop_duplicates(subset=['textTitle'])


news = news_unique['token_text']
news=list(news)

id2word = corpora.Dictionary(news)
corpus = [id2word.doc2bow(txt) for txt in news]
print(corpus[:5])

# down-weighing frequent words with tfidf
tfidf = models.TfidfModel(corpus, smartirs='ntc')
for doc in tfidf[corpus]:
    print([[id2word[id], np.around(freq, decimals=2)] for id, freq in doc])

corpus_tfidf = tfidf[corpus]
print(corpus_tfidf)
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus_tfidf[99:100]]

'''
--------------- EVALUATING LDA MODELS -------------------------------------------------------------
'''
import random
# evaluating topic models by calculating perplexity and likelihood
# for different topic numbers - manually, because none of the codes work.


lda_news10 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=10)
lda_news10.show_topics(10,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news10.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news14 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=14)
lda_news14.show_topics(14,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news14.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news20 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=20)
lda_news20.show_topics(20,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news20.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news25 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=25)
lda_news14.show_topics(25,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news25.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news30 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=30)
lda_news30.show_topics(30,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news30.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news40 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=40)
lda_news40.show_topics(40,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news40.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news50 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=50)
lda_news50.show_topics(50,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news50.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news100 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=100)
lda_news100.show_topics(100,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news100.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_news200 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=200)
lda_news200.show_topics(20,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_news200.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.


for idx, topic in lda_news14.print_topics(num_topics=200):
    print('Topic: {} Word: {}'.format(idx, topic))

'''
---------------- LDA News: CHOSEN MODEL ---------------------------------------------------------
We can utilize Latent Dirichlet Allocation (LDA), since the implementation in Gensim is straightforward. As before, we need to create a Dictionary and a Corpus, set the number of topics we want to infer and then finally associated a number of keywords for each topic.

HDP is an implementation of LDA, but the latter lets you infer the distributions while during HDP this inference is integrated in the model without any a priori knowledge of topics

In the example below, we set =5 topics and =10 keywords
'''

num_topics = 14

''' How to interpret this?

Topic 0 is a represented as _0.016“car” + 0.014“power” + 0.010“light” + 0.009“drive” + 0.007“mount” + 0.007“controller” + 0.007“cool” + 0.007“engine” + 0.007“back” + ‘0.006“turn”.

It means the top 10 keywords that contribute to this topic are: ‘car’, ‘power’, ‘light’.. and so on and the weight of ‘car’ on topic 0 is 0.016.

The weights reflect how important a keyword is to that topic.'''

lda_news = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=num_topics)
lda_news.show_topics(num_topics,10) # show topics with 10 keywords

#printing the topics of the model
pprint(lda_news.print_topics())
doc_lda_news = lda_news[corpus_tfidf]


# Compute Perplexity
''' 
It captures how surprised a model is of new data it has not seen before, and is measured as the normalized log-likelihood of a held-out test set
'''
print('\nPerplexity: ', lda_news.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

data_lemmatized = news_unique['cl_textTitle']
# Compute Coherence Score
coherence_model_lda_news = CoherenceModel(model=lda_news, texts=news, dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda_news.get_coherence()
print('\nCoherence Score: ', coherence_lda)

'''
----------- SAVING MODEL ----------------------------------------------------------------------------------
'''
#lda_news.save('lda_news.model')

# later on, load trained model from file
lda_news =  LdaModel.load('lda_news.model')
lda_news.show_topics(num_topics=14)

'''
-------------- NEWS dominant topics -----------------------------------------------
The below code extracts this dominant topic for each sentence and shows 
the weight of the topic and the keywords in a nicely formatted output 
'''

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=news):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_news[corpus]):
        row = row_list[0] if lda_news.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_news.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = pd.DataFrame()
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_news, corpus=corpus_tfidf, texts=news)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

''' 
## Finding the most representative document per topic to facilitate
## interpretation

'''

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(30)

# plotting the table

sent_topics_sorteddf_mallet.to_excel('news_topics.xlsx')

news_unique.to_excel('data_news.xlsx')


'''
What are the most discussed topics in the documents

Let’s make two plots:

1. The number of documents for each topic by assigning the document to the
  topic that has the most weight in that document.
2. The number of documents for each topic by by summing up the actual 
    weight contribution of each topic to respective documents.
'''
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs = model[corp]
        wordid_topics = model[corp]
        wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_news, corpus=corpus_tfidf, end=-1) 

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 10 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_news.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)

from matplotlib.ticker import FuncFormatter

df_dominant_topic_in_each_doc['perc']=df_dominant_topic_in_each_doc['count']/2695*100
# Plot


# Plot
fig, (ax1) = plt.subplots(1, figsize=(15, 6), dpi=120, sharey=True)
fig.suptitle('Number of News Documents by Topic', fontsize=16)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='perc', data=df_dominant_topic_in_each_doc, width=.4, color='powderblue')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x))
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_ylabel('% of Documents')
ax1.set_ylim(0, 15)

plt.show()


df_dominant_topic_in_each_doc = df_dominant_topic_in_each_doc.rename(columns={'Dominant_Topic':'Topic_Num'})
topics_news = pd.merge(sent_topics_sorteddf_mallet, df_dominant_topic_in_each_doc, on='Topic_Num')

topics_news.to_excel('topics_news.xlsx')



###################################################################################
###################################################################################
###################################################################################
###################################################################################
'''
-------------- CLEAN TWITTER DATA ------------------------------------------------
'''
# removing urls in tweets
data['clean_twt'] = [re.sub(r"((?:https?:\/\/(?:www\.)?|(?:pic\.|www\.)(?:\S*\.))(?:\S*))",'', x) for x in data['text']]


my_punctuation = r'!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•…' #punctuation
STOPWORDS = STOPWORDS
#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
wordnet_lemmatizer = WordNetLemmatizer()


def clean_tweet(tweet, bigrams=False, lemma=False):
    tweet = tweet.lower() # lower case
    tweet = re.sub(r'[^\w\s]', ' ', tweet) # strip punctuation
    tweet = re.sub(r'\s+', ' ', tweet) #remove double spacing
    tweet = re.sub(r'([0-9]+)', '', tweet) # remove numbers
    tweet = re.sub(r'([\U00002024-\U00002026]+)', '', tweet) #removing html tag ("..." where a link used to be)
    tweet_token_list = [word for word in tweet.split(' ')
                              if word not in STOPWORDS] # remove stopwords

    if lemma == True:
      tweet_token_list = [wordnet_lemmatizer.lemmatize(word) if '#' not in word else word
                        for word in tweet_token_list] # apply lemmatizer
    else:   # or                 
      tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

#Finally we apply the function to clean tweets (here we use lemmas)
data['clean_twt'] = data.clean_twt.apply(clean_tweet, lemma=True, bigrams=True)

data.head(5)

#We create a new column with tokens
data['token_tweet'] = [
    [word for word in tweet.split()  if word not in STOPWORDS]
    for tweet in data['clean_twt']]
print(data['token_tweet'])

# only english tweets

#data.to_csv('cleaned_data.csv')

data.to_pickle('data.pkl')
data = pd.read_pickle('data.pkl')
'''
---------------- tweet duplicates ------------------------------------------------
'''
# extra column with tweet length
data['tweet_length']  = data['clean_twt'].str.len() #based on length
#And let's take out all tweets that are more than 50 characters after cleaning
dflong = data[data.tweet_length > 50]

#Let's separate the duplicates

data.sort_values("clean_twt", inplace = True) 
duplicate_tweet = data[data.duplicated(['clean_twt'],keep=False)]
#how many
len(duplicate_tweet)
#how many unique tweets
len(duplicate_tweet.clean_twt.unique())
# how many times do they occur
duplicate_tweet['count'] = duplicate_tweet.groupby('clean_twt')['clean_twt'].transform('count')

#Let's see what they are saying:
''' To consider: should i only keep one example of the different tweets? what would that mean?
what about retweets?
'''

#We start by iterating through each unique duplicate
for n in duplicate_tweet.clean_twt.unique():
  #Here we use a bit of a trick: it's easier to read a tweet than a cleaned tweet
  #So let's just locate the corresponding tweet for each unique cleaned tweet
  print(duplicate_tweet.loc[duplicate_tweet['clean_twt'] == n, 'text'].iloc[0],
  #Wouldn't be nice to see the count too?
        " : ",
        duplicate_tweet.loc[duplicate_tweet['clean_twt'] == n, 'count'].iloc[0])

dflong.drop_duplicates(subset ="clean_twt", keep = False, inplace = True) 

dfshort = data[data.tweet_length <= 50]
df = pd.concat([dfshort, dflong], ignore_index=True)

'''
---------------- LDA tweet ---------------------------------------------------------
We can utilize Latent Dirichlet Allocation (LDA), since the implementation in Gensim is straightforward. As before, we need to create a Dictionary and a Corpus, set the number of topics we want to infer and then finally associated a number of keywords for each topic.

HDP is an implementation of LDA, but the latter lets you infer the distributions while during HDP this inference is integrated in the model without any a priori knowledge of topics

In the example below, we set =5 topics and =10 keywords
'''
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
from pprint import pprint
from collections import OrderedDict
import seaborn as sns
from gensim.models import CoherenceModel

'''
------------ BUILDING CORPUS + DICTIONARY -----------------------------------------------------------------
'''

twts = df['token_tweet']

id2word = corpora.Dictionary(twts)
corpus = [id2word.doc2bow(text) for text in twts]
print(corpus[556:560])

tfidf = models.TfidfModel(corpus, smartirs='ntc')
for doc in tfidf[corpus]:
    print([[id2word[id], np.around(freq, decimals=2)] for id, freq in doc])

corpus_tfidf = tfidf[corpus]
print(corpus_tfidf)
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus_tfidf[:77]]


'''
------------- EVALUATING MODELS ------------------------------------------------------------
'''

lda_twt10 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=10)
lda_twt10.show_topics(10,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt10.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt14 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=14)
lda_twt14.show_topics(14,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt14.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt20 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=20)
lda_twt20.show_topics(20,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt20.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt25 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=25)
lda_twt14.show_topics(25,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt25.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt30 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=30)
lda_twt30.show_topics(30,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt30.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt40 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=40)
lda_twt40.show_topics(40,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt40.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt50 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=50)
lda_twt50.show_topics(50,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt50.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt100 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=100)
lda_twt100.show_topics(100,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt100.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

lda_twt200 = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=200)
lda_twt200.show_topics(20,10) # show topics with 10 keywords
print('\nPerplexity: ', lda_twt200.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.


for idx, topic in lda_twt30.print_topics(num_topics=200):
    print('Topic: {} Word: {}'.format(idx, topic))


'''
------------- CHOSEN MODEL ---------------------------------------------------------------------
'''
total_topics = 20

''' How to interpret this?

Topic 0 is a represented as _0.016“car” + 0.014“power” + 0.010“light” + 0.009“drive” + 0.007“mount” + 0.007“controller” + 0.007“cool” + 0.007“engine” + 0.007“back” + ‘0.006“turn”.

It means the top 10 keywords that contribute to this topic are: ‘car’, ‘power’, ‘light’.. and so on and the weight of ‘car’ on topic 0 is 0.016.

The weights reflect how important a keyword is to that topic.'''

lda_tweet = models.LdaModel(corpus_tfidf, id2word=id2word, num_topics=total_topics)
lda_tweet.show_topics(total_topics,10)

#printing the topics of the model
pprint(lda_tweet.print_topics())
doc_lda_tweet = lda_tweet[corpus_tfidf]


# heatmap
df_lda_tweet = {i: OrderedDict(lda_tweet.show_topic(i,25)) for i in range(total_topics)}
#df_lda_tweet
df_lda_tweet = pd.DataFrame(df_lda_tweet)
df_lda_tweet = df_lda_tweet.fillna(0).T
print(df_lda_tweet.shape)

g=sns.clustermap(df_lda_tweet.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=1, figsize=(10, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()


import pyLDAvis.gensim

pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda_tweet, corpus_tfidf, id2word, mds='TSNE')
panel


# Compute Perplexity
''' It captures how surprised a model is of new data it has not seen before, and is measured as the normalized log-likelihood of a held-out test set
'''
print('\nPerplexity: ', lda_tweet.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

df_lemmatized = df['clean_twt']
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_tweet, texts=df_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

'''
------------- SAVE MODEL ------------------------------------------------------------------
'''
#lda_tweet.save('lda_tweet.model')

# later on, load trained model from file
lda_tweet =  LdaModel.load('lda_tweet.model')
lda_tweet.show_topics(num_topics=20)

'''
-------------- TWEETS dominant topics -----------------------------------------------


extracts dominant topic for each sentence and shows 
the weight of the topic and the keywords in a nicely formatted output 
'''

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=twts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_tweet[corpus]):
        row = row_list[0] if lda_tweet.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_tweet.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_tweet, corpus=corpus_tfidf, texts=twts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

''' 
This code gets the most exemplar sentence for each topic.
'''

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(30)

# plotting the table

sent_topics_sorteddf_mallet.to_excel('tweet_topics.xlsx')

'''
What are the most discussed topics in the documents

Let’s make two plots:

1. The number of documents for each topic by assigning the document to the
  topic that has the most weight in that document.
2. The number of documents for each topic by by summing up the actual 
    weight contribution of each topic to respective documents.
'''
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs = model[corp]
        wordid_topics = model[corp]
        wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_tweet, corpus=corpus, end=-1) 

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_tweet.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)

from matplotlib.ticker import FuncFormatter

df_dominant_topic_in_each_doc['perc']=df_dominant_topic_in_each_doc['count']/11068*100




# Plot
fig, (ax1) = plt.subplots(1, figsize=(15, 6), dpi=120, sharey=True)
fig.suptitle('Number of Twitter Documents by topic', fontsize=16)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='perc', data=df_dominant_topic_in_each_doc, width=.4, color='powderblue')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'T ' + str(x))
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('% of Documents')
ax1.set_ylim(0, 35)

plt.show()



df_dominant_topic_in_each_doc = df_dominant_topic_in_each_doc.rename(columns={'Dominant_Topic':'Topic_Num'})
topics_tweet = pd.merge(sent_topics_sorteddf_mallet, df_dominant_topic_in_each_doc, on='Topic_Num')

topics_tweet.to_excel('topics_tweet.xlsx')

'''
---------------- COMPARING MODELS WITHIN AND ACROSS --------------------------------
'''

import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode()

def plot_difference(mdiff, title="", annotation=None):
    """
    Helper function for plot difference between models
    """
    annotation_html = None
    if annotation is not None:
        annotation_html = [["+++ {}<br>--- {}".format(", ".join(int_tokens), 
                                              ", ".join(diff_tokens)) 
                            for (int_tokens, diff_tokens) in row] 
                           for row in annotation]
        
    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title,
                       xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    py.iplot(dict(data=[data], layout=layout))

# twitter model against news model
mdiff, annotation = lda_tweet.diff(lda_news, distance='jaccard', num_words=50)
plot_difference(mdiff, title="Topic difference (two models)[jaccard distance]", annotation=annotation)

# news model against itself
mdiff, annotation = lda_news.diff(lda_news, distance='jaccard', num_words=50)
plot_difference(mdiff, title="Topic difference (two models)[jaccard distance]", annotation=annotation)

# twitter model against itself
mdiff, annotation = lda_tweet.diff(lda_tweet, distance='jaccard', num_words=50)
plot_difference(mdiff, title="Topic difference (two models)[jaccard distance]", annotation=annotation)


