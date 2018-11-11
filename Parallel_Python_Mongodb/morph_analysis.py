import pandas as pd
import numpy as np
import warnings
import setcsv as sc
from konlpy.tag import Twitter
from konlpy.utils import pprint
from bson.son import SON
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from multiprocessing import Lock

warnings.filterwarnings("ignore", message="numpy.dtype size changed")


#round at fourth
def short_float(val):
    value = float("{:.4f}".format(val))
    return value



#tokenizer
def token(raw, pos=["Noun","Alpha","Verb","Number","Adjective","KoreanParticle","Punctuation","Determiner","Adverb","Conjecton","Excalmation","Foreign"], stopword=[]):

    twitter = Twitter()
    
    return [word for word, tag in twitter.pos(
        raw,
        norm=True,
        stem=True)
        
        if len(word)>=1 and tag in pos and word not in stopword
        ]

    
vectorize = TfidfVectorizer(tokenizer=token, sublinear_tf=True)


#apply tf-idf to reviews -> vectorize
def tfidf(desc):
    X = vectorize.fit_transform(desc)
    #print('fit_transform, (No.review {0}, features {1})'.format(X.shape[0],X.shape[1]))
    #features = vectorize.get_feature_names()
    #print(pd.DataFrame(data=X.toarray(), columns=features))
    vector_array = X.toarray()
    return vector_array


#compute cosine_similarity
def similarity(vector_arr, desc):
    sm=[] #result

    for i in range(len(desc)-1):
        srch_vector=vectorize.transform([desc[i]])
        for j in range(i+1, len(desc)):
            cosine_similar = cosine_similarity(srch_vector, [vector_arr[j]]).flatten()
            cosine_similar = short_float(float(cosine_similar))
            sm.append(cosine_similar)
    return sm



#q1, mid, q3, max per cId
def analysis(group):
    #max_list = []
    #mid_list = []
    #q1_list = []
    #q3_list = []

    g = group.groupby('cId')
    g_names = g.groups.keys()  #cId_names list


    #per cId
    for i in g_names:
        #morphs per cId
        group_cId = pd.DataFrame(g.get_group(i))
        reviews_cId = group_cId['desc'].tolist()        

        #similarity per cId
        vector_array = tfidf(reviews_cId)
        sm = similarity(vector_array,reviews_cId)

        #result
        result_max = print_want_val(sm, lambda x: np.percentile(sm,100))
        #max_list.append(result_max)
        #print("max: ", result_max)

        result_mid = print_want_val(sm, lambda x: np.percentile(sm,50))
        #mid_list.append(result_mid)
        #print("mid: ", result_mid)

        result_q1 = print_want_val(sm, lambda x: np.percentile(sm,25))
        #q1_list.append(result_q1)
        #print("1st quartile: ", result_q1)

        result_q3 = print_want_val(sm, lambda x: np.percentile(sm,75))
        #q3_list.append(result_q3)
        #print("3rd quartile: ", result_q3)

        #print ("**************************************")

'''
    df = pd.DataFrame(columns=['cId','_max','q3', 'mid','q1'])
    df['cId'] = g_names
    df['_max'] = max_list
    df['q3'] = q3_list
    df['mid'] = mid_list
    df['q1'] = q1_list
    
    #print df
    col1 = df._max.quantile([0.75]).tolist()
    col2 = df.q3.quantile([0.75]).tolist()
    col3 = df.mid.quantile([0.75]).tolist()
    col4 = df.q1.quantile([0.75]).tolist()

    
    t_cid = []
    for i in range(len(df)):
        if((df['_max'][i] >= col1[0]) and (df['q3'][i] >= col2[0]) and (df['mid'][i] >= col3[0]) and (df['q1'][i]>= col4[0])):
            t_cid.append(df['cId'][i])

    #print t_cid
    print len(t_cid)
    sc.setcsv(t_cid)
    
'''


def print_want_val(sm_list, used_func):
    val=(used_func(sm_list))
    return val






