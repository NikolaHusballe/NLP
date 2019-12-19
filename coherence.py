import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
from pprint import pprint
from collections import OrderedDict
import seaborn as sns
from gensim.models import CoherenceModel

# only using unique headlines

news_Unique = data.drop_duplicates(subset=['textTitle'])


news = news_Unique['token_text']

id2word = corpora.Dictionary(news)
corpus = [id2word.doc2bow(txt) for txt in news]
print(corpus[:5])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[99:100]]

total_topics = 7

''' How to interpret this?

The weights reflect how important a keyword is to that topic.'''

lda_news = models.LdaModel(corpus, id2word=id2word, num_topics=total_topics)
lda_news.show_topics(total_topics,10) # show topics with 10 keywords

#printing the topics of the model
pprint(lda_news.print_topics())
doc_lda_news = lda_news[corpus]

# heatmap
data_lda_news = {i: OrderedDict(lda_news.show_topic(i,25)) for i in range(total_topics)}
#data_lda
data_lda_news = pd.DataFrame(data_lda_news)
data_lda_news = data_lda_news.fillna(0).T
print(data_lda_news.shape)

g=sns.clustermap(data_lda_news.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=1, figsize=(10, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()


# Compute Perplexity
''' It captures how surprised a model is of new data it has not seen before, and is measured as the normalized log-likelihood of a held-out test set
'''
print('\nPerplexity: ', lda_news.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

data_lemmatized = news_Unique['cl_textTitle']
# Compute Coherence Score
coherence_model_lda_news = CoherenceModel(model=lda_news, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda_news.get_coherence()
print('\nCoherence Score: ', coherence_lda)
