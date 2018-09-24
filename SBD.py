# author: Bingqing Hu
from __future__ import division
from __future__ import print_function

import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

file_train = open(sys.argv[1], 'r') 
file_test = open(sys.argv[2],'r')

def clean_data(file):
    list = file.readlines()
    length = len(list)
    list = [i.split() for i in list]
    clean_list = []
    tok_list = []
    for j in range(length-1):
        if list[j][2] == 'EOS' or list[j][2] == 'NEOS':
            seq_num = list[j][0]
            label = list[j][2]
            L = list[j][1]
            R = list[j+1][1]
            clean_list.append([seq_num,L,L[0:-1],R,len(L)-1<3,L[0].isupper(),R[0].isupper(), \
            R=='"',len(L)==1, L.isdigit(), label])
        else:
            tok_list.append(list[j])

    clean_frame = pd.DataFrame(clean_list)
    tok_frame = pd.DataFrame(tok_list,columns=['s1','s2','s3'])

    return clean_frame, tok_frame

df_train,_ = clean_data(file_train)
df_test,tok_test = clean_data(file_test)
seq_test = df_test.values[:,0]
reserved_col = df_test.values[:,1]

X_train = df_train.iloc[:,2:-1]
X_test = df_test.iloc[:,2:-1]

# one-hot encoder 
df_merged = pd.concat([X_train,X_test])

enc = preprocessing.LabelEncoder()
df_merged = df_merged.apply(enc.fit_transform)
ohe = preprocessing.OneHotEncoder()
df_merged = ohe.fit_transform(df_merged)

X_train = df_merged[0:len(X_train),:]
X_test = df_merged[-len(X_test):-1,:]
y_train = df_train.values[:,-1]
y_test = df_test.values[:,-1]

# decision tree classifier
gini = DecisionTreeClassifier()
gini.fit(X_train,y_train)
y_pred = gini.predict(X_test)

accuracy = sum(1 for x,y in zip(y_test,y_pred) if x == y) / len(y_test)
print('accuracy:'+str(accuracy))

seq_test = pd.Series(seq_test)
reserved_col = pd.Series(reserved_col)
y_pred = pd.Series(y_pred)
df_out = pd.DataFrame(data=dict(s1=seq_test,s2=reserved_col,s3=y_pred))
df_out = pd.concat([df_out,tok_test])
df_out = df_out.sort_values(by='s1')
df_out.to_csv('SBD.test.out', header=None, index=None, sep=' ',mode='a')