#author: Bingqing Hu
from __future__ import print_function
from __future__ import division

import sys
from collections import Counter
import pandas as pd
import numpy as np
import math

file = open(sys.argv[1],'r')
corpus =  file.read()
corpus = corpus.split()
punctuation = [',','.','"','%','$',';','?']
corpus = list(filter(lambda a: a not in punctuation, corpus))
bigram_set = []
for i in range(len(corpus)-1):
    bigram_set.append((corpus[i],corpus[i+1]))

count_unigram = Counter(corpus)
count_bigram = Counter(bigram_set)

for key in list(count_bigram.keys()):
    if count_bigram[key] < 5:
        del count_bigram[key]

df_bigram = pd.DataFrame.from_dict(count_bigram,orient='index')
df_bigram.columns = ['bi_count']

L = [i[0] for i in df_bigram.index]
R = [i[1] for i in df_bigram.index]
L_count = [count_unigram[i] for i in L]
R_count = [count_unigram[i] for i in R]

df_bigram['L_count'] = L_count
df_bigram['R_count'] = R_count

N = len(corpus)

def chi_square_score(df,N):
    print(N)
    df['O12'] = df['L_count']-df['bi_count']
    df['O21'] = df['R_count']-df['bi_count']
    df['O22'] = N-df['bi_count']
    df['chi^2'] = (df['bi_count']*df['O22']-df['O12']*df['O21'])**2 \
    /(df['bi_count']+df['O12'])/(df['bi_count']+df['O21'])/(df['O22']+df['O12'])/(df['O22']+df['O21']) *N
    return df['chi^2'].sort_values(ascending=False).iloc[0:20]

def pmi_score(df,N):
    df['pmi'] = np.log(N*df['bi_count']/df['L_count']/df['R_count'])
    return df['pmi'].sort_values(ascending=False).iloc[0:20]

if sys.argv[2] == 'chi-square':
    result = chi_square_score(df_bigram,N)
elif sys.argv[2] == 'PMI':
    result = pmi_score(df_bigram,N)
else:
    print("method should either be 'chi-square' or 'pmi'!")

print(result)