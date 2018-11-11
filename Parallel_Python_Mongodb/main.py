import connect_mongodb as cm
import morph_analysis as ma
import timestamp as ts
import pandas as pd
import os
from multiprocessing import Process



def process_cursor(skip_n, limit_n):
    proc = os.getpid()
    print('Starting num: {0}, limit num: {1} by process id: {2}'.format(skip_n, limit_n, proc))


    collection = cm.connect() #connect to mongodb
    

    #each process access to certain cIds
    cursor = collection.find({u'cId':{'$in': cIds_list[skip_n:limit_n]}})
    #print(str(skip_n) + " mongodb read " + ts.timestamp())


    reviews = pd.DataFrame()
    for doc in cursor:
        tmp = pd.DataFrame(data=[[doc['cId'],doc['desc']]], columns=['cId','desc'])
        reviews = reviews.append(tmp, ignore_index=True)
        #print "cId: " + doc['cId'] + " desc: " + doc['desc']


    ma.analysis(reviews)
    
    print('Completed num: {0}, process: {1}'.format(skip_n, proc))




if __name__ == '__main__':
    print("start " + ts.timestamp())

    #connect to mongodb for creating cIds_list
    collection = cm.connect()


    #sorting by cIds except naverpay, count1
    pipeline = [{'$group':{'_id':'$cId', 'count':{'$sum':1}}}, {'$sort':{    'count':-1}}]
    cIds = pd.DataFrame(list(collection.aggregate(pipeline)))
    cIds = cIds.drop(cIds.index[0]) #drop naverpay
    cIds = cIds[cIds['count'] != 1] #drop count1
    cIds = cIds.sort_values(by=['_id'])
    cIds_list = cIds['_id'].tolist()
    del cIds    
    
    print ("mongodb access for preparing" + ts.timestamp())
    
    
    n_cores = 32
    collection_size = 151101 #num of cId : 151101

    #setting skips
    batch_size = int(round(collection_size/n_cores))
    skips = range(0, n_cores*batch_size, batch_size)


    #setting batchs
    limits=[]
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
        #print ("process start " + ts.timestamp())


    for process in processes:
        process.join()
        
print ("process finish " + ts.timestamp())










