import redis
import cPickle as pickle

r = redis.StrictRedis(host="localhost", port=6379, db=0)


def store_pickle(label, cucumber):
    cucumber_string = pickle.dumps(cucumber)
    r.set("doc2vec_" + label, cucumber_string)


def retrieve_pickle(label):
    return pickle.loads(r.get(label))


def retrieve_all_pickles():
    return [key for key in r.keys("doc2vec_*")]

