import os
import redis
import cPickle as pickle
import smart_open
import gensim

r = redis.StrictRedis(host="localhost", port=6379, db=0)
proj_root_dir = """C:\Users\jkerxhalli\Desktop\golf\doc2vec\\"""
files_dir = "TestAndTrainDocs"
data_files_dir = proj_root_dir + files_dir + "\\"


def check_data_files(filer_dir = ""):
    file_list = []
    for root, subdirs, files in os.walk(data_files_dir+filer_dir):
        [file_list.append(root.replace(data_files_dir, "") + "\\" + f) if root != data_files_dir else file_list.append(f) for f in files]
    return file_list


def concatenate_files(label, file_dir):
    files = check_data_files(file_dir)
    concat_file_path = os.path.join(data_files_dir + file_dir + "\\", label + ".cor")
    concat_file = open(concat_file_path, "a+")
    for f in files:
        lines = open(data_files_dir + f).readlines()
        [concat_file.write(line) if any(c.isalpha() for c in line) else "" for line in lines]
    return file_dir  + "\\" +  label + ".cor"

# new concat file? treat each full file as a line

def get_corpus_file(corpus_name):
    corpus_file = data_files_dir + corpus_name
    lines = open(corpus_file).readlines()
    return [line for line in lines]

def read_corpus(file,disallowWhitespace = False):
    # type: (File, bool) -> list[str]
    # ignore empty lines
    file = data_files_dir + file
    with smart_open.smart_open(file, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if any(c.isalpha() for c in line):
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])



def store_pickle(label, cucumber):
    cucumber_string = pickle.dumps(cucumber)
    r.set("doc2vec_" + label, cucumber_string)


def redis_retrieve(label):
    return r.get("doc2vec_" + label)


def retrieve_all_models():
    return [key.replace("doc2vec_",'') for key in r.keys("doc2vec_*")]

def redis_store(key, value):
    r.set("doc2vec_" + key, value)
