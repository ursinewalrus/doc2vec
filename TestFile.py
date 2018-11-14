import gensim
import os
import collections
import smart_open
import random
# https://github.com/RaRe-Technologies/gensim/blob/ca0dcaa1eca8b1764f6456adac5719309e0d8e6d/docs/notebooks/doc2vec-IMDB.ipynb
lee_train_file = "C:\Users\jkerxhalli\Desktop\golf\doc2vec\TestAndTrainDocs\lee.cor"
lee_test_file = "C:\Users\jkerxhalli\Desktop\golf\doc2vec\TestAndTrainDocs\lee_background.cor"


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, eopchs=10)
model.build_vocab(train_corpus)
model.train(documents=train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus) - 45):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])



str = "A young humpback whale remained tangled in a shark net off the Gold Coast yesterday, despite valiant efforts by marine rescuers. With its head snared by the net and an anchor rope wrapped around its tail, the stricken whale was still swimming but hopes for its survival were fading. A second rescue attempt was planned for dawn today after rescuers braved heavy seas, strong wind and driving rain to try to free the whale.Prince William has told friends his mother was right all along to suspect her former protection officer of spying on her and he doesn't want any detective intruding on his own privacy. William and Prince Harry are so devastated by the treachery of Ken Wharfe, whom they looked on as a surrogate father, they are now refusing to talk to their own detectives."
doc_id = random.randint(0, len(test_corpus))
inferred_vector = model.infer_vector(str.split(' '))
sims = model.docvecs.most_similar([inferred_vector])

print ' '.join(test_corpus[doc_id])
print sims[0][1]
print ' '.join(train_corpus[sims[0][0]].words)
print sims[1][1]
print ' '.join(train_corpus[sims[1][0]].words)