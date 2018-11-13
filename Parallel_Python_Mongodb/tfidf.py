#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from konlpy.tag import Twitter
from konlpy.utils import pprint
from bson.son import SON
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time
import os
from multiprocessing import Process
import pymongo
from pymongo import MongoClient



warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def connect():
    #connect to mongodb
    client = MongoClient('mongodb://addr')
    db = client['reviews']
    collection = db['reviews']
    return collection



def  token(raw, pos=["Noun","Alpha","Verb","Number","Adjective","KoreanParticle","Punctuation","Determiner","Adverb","Conjecton","Excalmation","Foreign"], stopword=[]):

    twitter = Twitter()

    return [word for word, tag in twitter.pos(
        raw,
        norm=True,
        stem=True)

        if len(word)>=1 and tag in pos and word not in stopword
        ]


vectorize = TfidfVectorizer(tokenizer=token, sublinear_tf=True)



#round at fourth
def short_float(val):
    value = float("{:.4f}".format(val))
    return value



#apply tf-idf to reviews -> vectorize
def tfidf(desc):
    
    X = vectorize.fit_transform(desc)
    print('fit_transform, (No.review {}, feature {})'.format(X.shape[0], X.shape[1]))
    features = vectorize.get_feature_names()
    #print(pd.DataFrame(data=X.toarray(), columns=features))
    vector_array = X.toarray()

    return vector_array



def similarity(vector_arr, desc):
    sm=[] # result
    
    for i in range(len(desc)-1):
        srch_vector=vectorize.transform([desc[i]])
        for j in range(i+1, len(desc)):
            cosine_similar = cosine_similarity(srch_vector, [vector_arr[j]]).flatten()
            cosine_similar = short_float(float(cosine_similar))
            sm.append(cosine_similar)
  
    return sm



def analysis(group):
    max_list=[]
    mid_list=[]
    q1_list=[]
    q3_list=[]

    g = group.groupby('cId')
    g_names = g.groups.keys()  #cId_names list
    
        
    #per cId
    for i in g_names:

        #morphs per cId
        group_cId = pd.DataFrame(g.get_group(i))
        reviews_cId = group_cId['desc'].tolist()

        #similarity per cId
        vector_array=tfidf(reviews_cId)
        sm = similarity(vector_array,reviews_cId)
        

        # result
        result_max = print_want_val(sm, lambda x: np.percentile(sm,100))
        print ("max: ", result_max)
        max_list.append(result_max)
            
        result_mid = print_want_val(sm, lambda x: np.percentile(sm,50))
        print ("mid: ", result_mid)
        mid_list.append(result_mid)

        result_q1 = print_want_val(sm, lambda x: np.percentile(sm,25))
        print ("1st quartile: ", result_q1)
        q1_list.append(result_q1)

        result_q3 = print_want_val(sm, lambda x: np.percentile(sm,75))
        print ("3rd quartile: ", result_q3)
        q3_list.append(result_q3)


        print ('*********************************')



def print_want_val(sm_list, used_func):
    val=(used_func(sm_list))
    return val



def process_cursor(skip_n, limit_n):
    proc = os.getpid()
    print('Starting num: {0}, limit num: {1} by process id: {2}'.format(skip_n, limit_n, proc))
    collection = connect()


    #cursor = collection.find({}).skip(skip_n).limit(limit_n)
    cursor = collection.find({u'cId':{'$in': cIds_list[skip_n:limit_n]}})
    

    reviews = pd.DataFrame()
    for doc in cursor:
        tmp = pd.DataFrame(data=[[doc['cId'],doc['desc']]], columns=['cId','desc'])
        reviews = reviews.append(tmp, ignore_index=True)
        #print "cId: " + doc['cId'] + " desc: " + doc['desc']


    reviews.columns = ['cId','desc']
    analysis(reviews)
    
    print('Completed num: {0}, process: {1}'.format(skip_n, proc))


'''
test = connect()
cursor = test.find({u'cId':u"'_'"})
reviews = []
for doc in cursor:
    reviews.append(doc['desc'])
print len(reviews)
#(num_of_rev, num_of_f, vector_array) = tfidf(reviews)
#similarity('*.*', num_of_rev, num_of_f, vector_array, reviews)
'''


#execution
if __name__ == '__main__':

    #connect to mongodb for creating cIds_list
    client = MongoClient('mongodb://ms:kmubigdata2018@203.246.113.16:6235/')
    db = client['reviews']
    collection = db['reviews']


    # sorting by cIds except naverpay, count1
    pipeline = [{'$group':{'_id':'$cId', 'count':{'$sum':1}}}, {'$sort':{'count':-1}}]
    cIds = pd.DataFrame(list(collection.aggregate(pipeline)))
    cIds = cIds.drop(cIds.index[0]) #drop naverpay
    cIds = cIds[cIds['count'] != 1] #drop count1   
    cIds = cIds.sort_values(by=['_id'])
    cIds_list = cIds['_id'].tolist()


    n_cores = 10
    collection_size = 100 # max: 151101

  
    #setting skips
    batch_size = round(collection_size/n_cores) 
    batch_size = int(batch_size)
    skips = range(0, n_cores*batch_size, batch_size)
    #skips = range(0, collection_size, batch_size)

	
    #setting batchs
    limits = []
    for i in range(1, n_cores):
    	limits.append(batch_size*i)
    
    if(collection_size%n_cores == 0):
    	limits.append(batch_size*n_cores)
    else:
    	limits.append(batch_size*n_cores + collection_size%n_cores)
    
    
    #create processes
    processes = [Process(target=process_cursor, args=(skip_n, limit_n)) for skip_n, limit_n in zip(skips, limits)]


    for process in processes:
    	process.start()

    for process in processes:
    	process.join()





