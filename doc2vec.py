import gensim
import os
import random
from doc2VecExplorer import redisHandler as rh, doc2vecModelUtils as d2vu, dataUtils as du

# Set file names for train and test data
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
# lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
# lee_test_file = test_data_dir + os.sep + 'lee.cor'
#
# print "start read"
# train_corpus = list(d2vu.read_corpus(lee_train_file))
# print "end read"
#
# test_corpus = list(d2vu.read_corpus(lee_test_file, tokens_only=True))
#
# model = gensim.models.doc2vec.Doc2Vec(vectorsize=300, min_count=2, epochs=5)
#
# print "start build"
# model.build_vocab(train_corpus)
# print "end build"
#
# print "start train"
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
# print "end train"
#
# print "start store"
# rh.store_pickle('lee',model)
# print "end store"
#
# user_input = raw_input("Get compare")
# while input != "exit":
#     doc_id = random.randint(0, len(test_corpus) - 1)
#     inferred_vector = model.infer_vector(test_corpus[doc_id])
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#
#     print "---"
#     print ' '.join(test_corpus[doc_id])
#     print sims[0][1]
#     print ' '.join(train_corpus[sims[0][0]].words)
#     print sims[1][1]
#     print ' '.join(train_corpus[sims[1][0]].words)
#     print "---"
#     user_input = raw_input("new selection\n")

command = ''

options = {
    '1': lambda: "\n".join(du.check_data_files()),
    '2': lambda: d2vu.get_train_files_args(False),
    '3': lambda: d2vu.get_train_files_args(True),
    '4': lambda: "\n".join(rh.retrieve_all_pickles()),
    '5': lambda: d2vu.load_main_model(),
    '6': lambda: d2vu.load_test_corpus(),
    '7': lambda: "\n".join(d2vu.test_loaded_corpus_against_loaded_model())
}

while command != 'exit':
    print """Commands:
    Check trainable files: 1
    Train model from file: 2 
    Train model from dir: 3 
    Check available models: 4
    Load trained model: 5 
    Load test corpus(trainable files): 6 <name>
    Test corpus against model: 7
    Test arbitrary phrase against model: 8 <text>
    """
    command = raw_input()
    if command in options:
        try:
            print options[command]() + "\n"
        except Exception as e:
            print e

