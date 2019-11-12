#%%
import matplotlib.pyplot as plt
%matplotlib inline
from to_trecrun import get_paragraph_transitions
import os
import numpy as np
from tqdm import tqdm
import math
import ujson as json
from run_info import RUN_IDS, RUN_INFO, RUN_INFO_BY_ID

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

# %%
run_id_to_transitions = {}

for run_id in tqdm(RUN_IDS):
    input_path = "data/car_runs/{0}.gz".format(run_id)
    run_id_to_transitions[run_id] = get_paragraph_transitions(input_path)

# %%
with open('data/transitions.json') as fp:
    transitions_list = json.load(fp)['labels']
#%%
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

query_to_transitions: DefaultDict[str, Dict[Tuple[str, str], int]] = defaultdict(dict)
for titem in transitions_list:
    qid = titem['query_id']
    para1_id = titem['para1_id']
    para2_id = titem['para2_id']
    label = titem['label']
    score = 0
    if label == 'AppropriateTransition' or label == 'SameTransition':
        score = 1
    query_to_transitions[qid][(para1_id, para2_id)] = score

#%%
query_ids = query_to_transitions.keys()

# %%
run_id_to_scores = {}

for run_id in RUN_IDS:
    transitions = run_id_to_transitions[run_id]
    scores = []
    for query in sorted(query_ids):
        if query not in transitions:
            scores.append(0.0)
        else:
            N = len(transitions[query])
            score = 0
            for para1_to_2 in transitions[query]:
                score += query_to_transitions[query].get(para1_to_2, 0)
            scores.append(score / N)
    run_id_to_scores[run_id] = np.array(scores)

# %%
transitions.keys(), run_id_to_transitions['UNH-bm25-rm'].keys()
#%%
boxplot_data = []

def run_is_bert(name):
    return RUN_INFO_BY_ID[name].get("bert", False)
def run_is_neural(name):
    info = RUN_INFO_BY_ID[name]
    return info.get("bert", False) or info.get("neural", False)

for run_id in RUN_IDS:
    measure_vec = run_id_to_scores[run_id]
    name = run_id
    if run_is_neural(run_id):
        name += '*'
    if run_is_bert(run_id):
        name += '#'
    boxplot_data.append((np.median(measure_vec), name, measure_vec))

# %%
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
plt.savefig('per-query-coherence.png', transparent=False, bbox_inches='tight', facecolor='white')
plt.show()

# %%
