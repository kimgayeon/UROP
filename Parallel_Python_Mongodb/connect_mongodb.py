import pymongo
from pymongo import MongoClient


#connect to pymongo
def connect():
    client = MongoClient('mongodb://ms:kmubigdata2018@203.246.113.16:6235/')
    db = client['reviews']
    collection = db['reviews']
    return collection

