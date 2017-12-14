import pandas as pd
import numpy as np
b_dir = '../kg-data/nation/'

entities = pd.read_csv(b_dir + 'entities.txt', sep='\t', header=None)
entities.columns = ['entity_id', 'entity_name']
relations = pd.read_csv(b_dir + 'relations.txt', sep='\t', header=None)
relations.columns = ['relation_id', 'relation_name']
triples = pd.read_csv(b_dir + 'triples.txt', sep='\t', header=None)
triples.columns = ['relation_id', 'entity_id1', 'entity_id2']

# https://github.com/mnick/scikit-kge
from skge import HolE, StochasticTrainer

# Load knowledge graph
# N = number of entities
# M = number of relations
# xs = list of (subject, object, predicte) triples
# ys = list of truth values for triples (1 = true, -1 = false)
N, M, xs, ys = len(entities), len(relations), np.array(triples[[
    'entity_id1', 'entity_id2', 'relation_id']]), np.ones(len(triples), dtype=np.int)

# instantiate HolE with an embedding space of size 100
model = HolE((N, N, M), 50)

# instantiate trainer
trainer = StochasticTrainer(model, nbatches=10,
                            max_epochs=50)

# fit model to knowledge graph
trainer.fit(xs, ys)

model.params['E'].shape
model.params['R'].shape

from scipy.spatial import distance


E = model.params['E']


def get_most_similar(e_id, N=1):
    # return distance.cdist([E[e_id]], np.delete(E, [e_id], axis=0)).argmin()
    return distance.cdist([E[e_id]], E).argsort()[0][1:N + 1]


N = 3
dic_E = entities.set_index('entity_id')['entity_name'].to_dict()
lst_similar = []
for e in sorted(dic_E.keys()):
    most_similar_N = [dic_E[e] for e in get_most_similar(e, N)]
    lst_similar.append(tuple([dic_E[e]] + most_similar_N))
df = pd.DataFrame(lst_similar)
df.columns = ['entity_name'] + ['most similar %s' %
                                str(i) for i in range(1, N + 1)]
from tastu_teche.plt_show import df_show
df_show(df, 'df_similar.txt', 'most_similar_N')


def save_kge_vectors(E, fname):
    with open(fname, 'w') as f:
        lst_line = []
        lst_line.append('%i %i' % E.shape)
        for line_no in range(E.shape[0]):
            lst_line.append('%s %s' % (
                dic_E[line_no], ' '.join([str(v) for v in E[line_no]])))
        f.write('\n'.join(lst_line))
