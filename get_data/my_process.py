import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import ipdb
from gensim.models.keyedvectors import KeyedVectors


# build data
def build_data( df ):
    train = []
    dev = []
    test = []
    vocab = defaultdict(float)

    print(df.columns)
    for idx, (label, sent, split) in df.iterrows():
        words = sent.split()
        for word in words:
            vocab[word] += 1
        datum = { "y": label, "text": sent }
        if split == 'train':
            train.append( datum )
        elif split == 'dev':
            dev.append( datum )
        elif split == 'test':
            test.append( datum )
    return train, dev, test, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def get_W2(word_vecs1, word_vecs2, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs1)
    word_idx_map = dict()
    W1 = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W2 = np.zeros(shape=(vocab_size+1, k), dtype='float32')

    W1[0] = np.zeros(k, dtype='float32')
    W2[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs1:
        W1[i] = word_vecs1[word]
        W2[i] = word_vecs2[word]
        word_idx_map[word] = i
        i += 1
    return W1, W2, word_idx_map

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__=="__main__":

    w2v_file = 'google_word2vec.bin'
    glove_file = 'glove_word2vec.txt'


    print( "loading google word2vec vectors..." )
    model1 = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    print(  "word2vec loaded!\n" )

    print( "loading glove word2vec vectors..." )
    model2 = KeyedVectors.load_word2vec_format(glove_file, binary=False)
    print(  "word2vec loaded!" )

    for dataset in ['MR', 'SST1', 'SST2']:
        print( "loading data...", dataset )
        df = pd.read_pickle('{}.pkl'.format(dataset))
        train, dev, test, vocab = build_data(df)
        assert len(df) == len(train) + len(test) + len(dev)
        print( "data loaded!" )
        print( "number of sentences: " + str(len(df)))
        print( "vocab size: " + str(len(vocab)) )

        google_w2v = dict( (w, model1.wv[w]) for w in vocab if w in model1.wv )
        glove_w2v = dict( (w, model2.wv[w]) for w in vocab if w in model2.wv )

        print(len(google_w2v), ' already in word2vec')
        print(len(glove_w2v), ' already in glove')

        add_unknown_words(google_w2v, vocab)
        add_unknown_words(glove_w2v, vocab)

        google_W, glove_W, word_idx_map = get_W2(google_w2v, glove_w2v)

        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        random_W, _ = get_W(rand_vecs)

        pickle.dump([train, dev, test, google_W, random_W, glove_W, word_idx_map, vocab], open("{}.p".format(dataset), "wb"))
