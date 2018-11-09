import gensim
import smart_open
import redisHandler as rh
import dataUtils as du
import random
# yield gensim.utils.simple_preprocess(line)


mainModel = None
testCorpus = None


def load_main_model():
    global mainModel
    label = raw_input("Select Model\n")
    mainModel = rh.retrieve_pickle(label)
    return "Set " + label + " to the active model"


def load_test_corpus():
    global testCorpus
    label = raw_input("Select Corpus\n")
    testCorpus = du.get_corpus_file(label)
    return "Loaded " + label + " as test corpus"

# when we make the model we need to also save the original text for indicing
# https://markroxor.github.io/gensim/static/notebooks/doc2vec-lee.html
def test_loaded_corpus_against_loaded_model():
    if mainModel is None:
        return ["Load a model first"]
    if testCorpus is None:
        return ["Load a corpus first"]
    tests_to_run = int(raw_input("Input number of tests to run"))
    for i in range(0,tests_to_run):
        selection = random.randint(0,len(testCorpus)-1)
        inferred_vector = mainModel.infer_vector(testCorpus[selection])
        sims = mainModel.docvecs.most_similar([inferred_vector], topn=len(mainModel.docvecs))
        print ' '.join(testCorpus[selection])
    return ["Compare Done"]


def read_corpus(file):
    # type: (File, bool) -> list[str]
    with smart_open.smart_open(file, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def train_model_from_file_data(label, file_name, options):
    if options == [[]]:
        options = [300, 5, 5]
    parsed_corpus = list(read_corpus(du.data_files_dir + file_name))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=options[0], min_count=options[1], epochs=options[2])
    model.build_vocab(parsed_corpus)
    model.train(parsed_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    rh.store_pickle(label, model)
    return "Trained " + label + "model from file " + file_name


def assess_model(model, corpus):
    # type: (gensim.models.doc2vec.Doc2Vec,list[str]) -> list[tuple]
    ranks = []
    second_ranks = []
    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    return "Compare Done"


def get_train_files_args(multipleFiles):
    input_args = raw_input("<model name> "
                           "<file name/dir> "
                           "<optional:vector size> "
                           "<optional:min word count> "
                           "<optional:epochs>\n")
    arguments = []
    arguments = input_args.split(' ')
    if multipleFiles:
        du.concatenate_files(arguments[0], arguments[1])
        arguments[1] = arguments[1] + "\\" + arguments[0] + '.cor'
    if len(arguments) < 3:
        arguments.append([])
    return train_model_from_file_data(arguments[0], arguments[1], arguments[2:])

