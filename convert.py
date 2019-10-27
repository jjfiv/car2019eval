#%%
import matplotlib.pyplot as plt
%matplotlib inline
from to_trecrun import convert_to_section_trecrun
from invoke_treceval import execute_treceval
import os
import numpy as np
from tqdm import tqdm

RUN_IDS = """
Bert-ConvKNRM-50
Bert-ConvKNRM
Bert-DRMMTKS
ECNU_BM25
ECNU_BM25_1
ECNU_ReRank1
ICT-BM25
ICT-DRMMTKS
IRIT_run1
IRIT_run2
IRIT_run3
ReRnak2_BERT
ReRnak3_BERT
UNH-bm25-ecmpsg
UNH-bm25-rm
UNH-bm25-stem
UNH-dl100
UNH-dl300
UNH-ecn
UNH-qee
UNH-tfidf-lem
UNH-tfidf-ptsim
UNH-tfidf-stem
UvABM25RM3
UvABottomUp1
UvABottomUp2
UvABottomUpChangeOrder
bm25-populated
dangnt-nlp
""".strip().split()

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
measure = "map"
boxplot_data = []

for run_id in RUN_IDS:
    data = run_id_to_measures[run_id]
    measure_vec = np.array([data[qid].get(measure, 0.0) for qid in sorted(query_ids)])
    boxplot_data.append((np.median(measure_vec), run_id, measure_vec))

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
plt.boxplot(xs,labels=labels,showmeans=True,meanline=True,notch=False)
plt.xticks(rotation=90,color='black')
plt.savefig('per-heading-{0}.png'.format(measure), transparent=False, bbox_inches='tight', facecolor='white')
plt.show()


#%%
