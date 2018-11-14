import gensim
import dataUtils as du
import random
import collections

mainModel = None
mainModelCorpus = None
testCorpus = None


def load_main_model():
    global mainModel
    global mainModelCorpus
    label = raw_input("Select Model\n")
    mainModel = gensim.models.Word2Vec.load(".\Models\\"+label)
    # mainModel = du.retrieve_pickle(label)
    mainModelCorpus = list(du.read_corpus(du.redis_retrieve(label)))#du.retrieve_pickle(label + "_corpus")
    return "Set " + label + " to the active model"


def load_test_corpus():
    global testCorpus
    label = raw_input("Select Corpus\n")
    testCorpus = list(du.read_corpus(label))
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
        try:
            selection = random.randint(0,len(testCorpus))
            inferred_vector = mainModel.infer_vector(testCorpus[selection].words)
            sims = mainModel.docvecs.most_similar(positive=[inferred_vector], topn= len(testCorpus))
            print sims[0][1]
            print sims[0][0]

            test = ' '.join(testCorpus[selection].words)
            bestnum = str(sims[0][1])
            best = ' '.join(mainModelCorpus[sims[0][0]].words)
            secondbestnum = str(sims[1][1])
            secondbest = ' '.join(mainModelCorpus[sims[1][0]].words)
            print selection
            print "Test: " + test
            print "Best match similarity " + bestnum
            print "Best match: " + best
            print "Second best match similarity " + secondbestnum
            print "Second Best match: " + secondbest
            print "___"
        except IndexError as ie:
            print ie
    return ["Compare Done"]

def test_arbitrary_phrases():
    phrase = raw_input("Phrase to test\n")
    while phrase != 'exit':
        inferred_vector = mainModel.infer_vector(phrase.split(' '))
        sims = mainModel.docvecs.most_similar(positive=[inferred_vector])
        bestnum = str(sims[0][1])
        best = ' '.join(mainModelCorpus[sims[0][0]].words)
        secondbestnum = str(sims[1][1])
        secondbest = ' '.join(mainModelCorpus[sims[1][0]].words)
        print "Best match similarity " + bestnum
        print "Best match: " + best
        print "Second best match similarity " + secondbestnum
        print "Second Best match: " + secondbest
        print "___"
        phrase = raw_input("Phrase to test\n")


def train_model_from_file_data(label, file_name, options):
    global mainModel, mainModelCorpus, testCorpus
    if options == [[]]:
        options = [300, 2, 1]
    parsed_corpus = list(du.read_corpus(file_name))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=int(options[0]), min_count=int(options[1]), epochs=int(options[2]))
    model.build_vocab(parsed_corpus)
    model.train(parsed_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    mainModel = model
    mainModelCorpus = parsed_corpus
    mainModel.save(".\Models\\"+label)
    # testCorpus = list(du.read_corpus(file_name))
    du.redis_store(label, file_name)
    return "Trained " + label + " model from file " + file_name


def assess_model():
    # type: (gensim.models.doc2vec.Doc2Vec,list[str]) -> list[tuple]
    ranks = []
    second_ranks = []
    for doc_id in range(len(mainModelCorpus)):
        inferred_vector = mainModel.infer_vector(mainModelCorpus[doc_id].words)
        sims = mainModel.docvecs.most_similar([inferred_vector], topn=len(mainModel.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    print collections.Counter(ranks)
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
        arguments[1] = du.concatenate_files(arguments[0], arguments[1])
    if len(arguments) < 3:
        arguments.append([])
    return train_model_from_file_data(arguments[0], arguments[1], arguments[2:])

