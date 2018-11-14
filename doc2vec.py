import gensim
import os
import random
from doc2VecExplorer import doc2vecModelUtils as d2vu, dataUtils as du
# https://radimrehurek.com/gensim/apiref.html
# make a config file geeze
command = ''

options = {
    '1': lambda: "\n".join(du.check_data_files()),
    '2': lambda: d2vu.get_train_files_args(False),
    '3': lambda: d2vu.get_train_files_args(True),
    '4': lambda: "\n".join(du.retrieve_all_models()),
    '5': lambda: d2vu.load_main_model(),
    '6': lambda: d2vu.load_test_corpus(),
    '7': lambda: "\n".join(d2vu.test_loaded_corpus_against_loaded_model()),
    '8': lambda: "\n".join(d2vu.test_arbitrary_phrases()),
    '9': lambda:  d2vu.assess_model()
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
    Test arbitrary phrases against model, 'exit' to return: 8 <text>
    Assess loaded model: 9
    """
    command = raw_input()
    if command in options:
        try:
            print options[command]() + "\n"
        except Exception as e:
            print e

