#%%
import matplotlib.pyplot as plt
%matplotlib inline
from to_trecrun import convert_to_section_trecrun
from invoke_treceval import execute_treceval
import os
import numpy as np
from tqdm import tqdm

RUN_INFO = [
{'id': 'Bert-ConvKNRM-50', 'bert': True},
{'id': 'Bert-ConvKNRM', 'bert': True},
{'id': 'Bert-DRMMTKS', 'bert': True},
{'id': 'ECNU_BM25', 'bert': False},
{'id': 'ECNU_BM25_1', 'bert': False},
{'id': 'ECNU_ReRank1', 'bert': False, 'neural': True},
{'id': 'ICT-BM25', 'neural': True},
{'id': 'ICT-DRMMTKS', 'neural': True},
{'id': 'IRIT_run1', 'neural': False},
{'id': 'IRIT_run2', 'neural': False},
{'id': 'IRIT_run3', 'neural': False},
{'id': 'ReRnak2_BERT', 'bert': True},
{'id': 'ReRnak3_BERT', 'bert': True},
{'id': 'UNH-bm25-ecmpsg'},
{'id': 'UNH-bm25-rm'},
{'id': 'UNH-bm25-stem'},
{'id': 'UNH-dl100', 'neural': True},
{'id': 'UNH-dl300', 'neural': True},
{'id': 'UNH-ecn'},
{'id': 'UNH-qee'},
# Malformed:
# {'id': 'neural', 'neural': True},
{'id': 'UNH-tfidf-lem'},
{'id': 'UNH-tfidf-ptsim'},
{'id': 'UNH-tfidf-stem'},
{'id': 'UvABM25RM3'},
{'id': 'UvABottomUp1'},
{'id': 'UvABottomUp2'},
{'id': 'UvABottomUpChangeOrder'},
{'id': 'bm25-populated'},
{'id': 'dangnt-nlp', 'bert': True},
]

RUN_INFO_BY_ID = dict((dat['id'], dat) for dat in RUN_INFO)
RUN_IDS = list(RUN_INFO_BY_ID.keys())

SECTION_QREL_FILE = "data/section.qrel"

query_ids = set()
with open(SECTION_QREL_FILE) as fp:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        qid = line.split()[0]
        query_ids.add(qid)

print("Using {0} query ids for comparison:".format(len(query_ids)))

#%%
run_id_to_measures = {}

for run_id in tqdm(RUN_IDS):
    input_path = "data/car_runs/{0}.gz".format(run_id)
    trecrun_path = "data/trecrun/{0}.section.trecrun".format(run_id)
    # convert from JSON to trecrun
    try:
        convert_to_section_trecrun(input_path, trecrun_path)
        assert os.path.exists(trecrun_path)
        runid, data = execute_treceval(SECTION_QREL_FILE, trecrun_path)
    except ValueError as e:
        raise ValueError("{0}: {1}".format(runid, e))
    run_id_to_measures[run_id] = data

#%%
measure = "ndcg"
boxplot_data = []

def run_is_bert(name):
    return RUN_INFO_BY_ID[name].get("bert", False)
def run_is_neural(name):
    info = RUN_INFO_BY_ID[name]
    return info.get("bert", False) or info.get("neural", False)

for run_id in RUN_IDS:
    data = run_id_to_measures[run_id]
    name = run_id
    if run_is_neural(run_id):
        name += '*'
    if run_is_bert(run_id):
        name += '#'
    measure_vec = np.array([data[qid].get(measure, 0.0) for qid in sorted(query_ids)])
    boxplot_data.append((np.median(measure_vec), name, measure_vec))

#%%

plt.style.use('ggplot')

best_to_worst = sorted(boxplot_data, reverse=True)
labels = [lbl for (_,lbl,_) in best_to_worst]
xs = [dat for (_,_,dat) in best_to_worst]

fig = plt.figure(figsize=(16,9),facecolor='white', dpi=150)
ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
#ax.set_facecolor('white')
#for pos in ['bottom', 'left', 'top', 'right']:
    #ax.spines[pos].set_color('black')
plt.boxplot(xs,labels=labels,meanline=True,notch=False)
plt.xticks(rotation=90, color='black')
plt.savefig('per-heading-{0}.png'.format(measure), transparent=False, bbox_inches='tight', facecolor='white')
plt.show()


#%%
