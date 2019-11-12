#%%
import matplotlib.pyplot as plt
%matplotlib inline
from to_trecrun import convert_to_section_trecrun
from invoke_treceval import execute_treceval
import os
import numpy as np
from tqdm import tqdm
import math
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

def pairwise(baseline, treatment):
    """
    Derived shamelessly from the Galago impl.
    """
    N = len(baseline)
    base_mean = np.mean(baseline)
    treat_mean = np.mean(treatment)
    difference = treat_mean - base_mean
    batch = 10000

    maxIterWithoutMatch = 1_000_000
    iterations = 0
    matches =  0

    left_sample = np.zeros(N)
    right_sample = np.zeros(N)

    p_value = 0.0
    while True:
        print(iterations)
        for i in range(batch):
            swaps = np.random.choice(a=[False, True], size=N)
            for j in range(N):
                if swaps[j]:
                    left_sample[j] = baseline[j]
                    right_sample[j] = treatment[j]
                else:
                    left_sample[j] = treatment[j]
                    right_sample[j] = baseline[j]
            sample_difference = np.mean(left_sample) - np.mean(right_sample)
            if difference <= sample_difference:
                matches += 1
        iterations += batch
        p_value = matches / iterations

        if matches == 0:
            if iterations < maxIterWithoutMatch:
                print("no-match-continue")
                continue
            else:
                break
        
        max_deviation = max(0.0000005 / p_value, min(0.00005 / p_value, 0.05))
        est_iter = math.sqrt(p_value * (1.0 - p_value)) / max_deviation
        if est_iter < iterations:
            print("est_iter", est_iter, "iterations", iterations)
            break
    return p_value


# %%
print(pairwise(xs[4], xs[3]))


# %%
