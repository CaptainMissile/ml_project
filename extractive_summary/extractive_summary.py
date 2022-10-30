import re
import numpy as np
import pandas as pd
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# function to remove stopwords
def remove_stopwords(sen):
  stop_words = stopwords.words('english')
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

def generate_extractive_summary(text):
    sentences = [sent_tokenize(text)]

    # flatten the list
    sentences = [y for x in sentences for y in x]

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((300,))
    sentence_vectors.append(v)

    # similarity matrix
    sim_mat =  np.zeros([1000, 1000]) # np.zeros([len(sentences), len(sentences)])

    

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]

    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    sn = 4
    # Generate summary
    for i in range(sn):
        print(ranked_sentences[i][1])



text = '''Bangladesh chose to bat. Match starts in 3 hrs 38 mins Match yet to begin Melbourne Renegades Women Sydney Sixers Women Match starts in 38 mins Day 2 Tasmania trail by 348 runs. Alphabetically sorted top ten of players who have played the most matches across formats in the last 12 months Also Known As When the annals of Bangladesh cricket are sifted by future generations Shakib Al Hasan will emerge and reemerge as the greatest cricketer of its first two decades. As a bowler Shakib is accurate consistent and canny aggression and a wide range of strokes are the keys to his batting. Even more importantly he has selfbelief and an excellent temperament unflustered by the big occasion and ready to do battle against the top teams.'''

generate_extractive_summary(text)