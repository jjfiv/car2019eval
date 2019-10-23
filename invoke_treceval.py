import os
import subprocess
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
import numpy as np
import sys

TREC_EVAL_BIN = os.getenv(
    "TRECEVAL", os.path.join(os.getenv("HOME", "~"), "bin", "trec_eval", "trec_eval")
)

assert os.path.exists(TREC_EVAL_BIN)
assert os.path.isfile(TREC_EVAL_BIN)


def execute_treceval(
    qrel: str, runfile: str
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    assert os.path.exists(qrel)
    assert os.path.isfile(qrel)
    assert os.path.exists(runfile)
    assert os.path.isfile(runfile)
    eval_cmd = [TREC_EVAL_BIN, "-q", "-m", "all_trec", qrel, runfile]
    completed = subprocess.run(eval_cmd, capture_output=True)
    if completed.returncode != 0:
        raise ValueError(
            "trec_eval: {0} for {2}; err: {1}".format(
                completed.returncode, completed.stderr.decode("utf-8"), runfile
            )
        )
    query_measure_score: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    runid = runfile
    for line in completed.stdout.decode("utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            [measure, query, score_str] = line.split()
            if measure == "runid":
                runid = score_str
                continue
            if query == "all":
                continue
            if measure == "relstring":
                continue
            score = float(score_str)
            query_measure_score[query][measure] = score
        except ValueError:
            print(line)
            sys.exit(-1)
    return runid, query_measure_score


if __name__ == "__main__":
    runid, data = execute_treceval("data/section.qrel", "data/UNH-bm25-ecmpsg.trecrun")
    distr = [q["ndcg"] for qid, q in data.items()]
    print(len(distr))
    print(np.mean(distr))
    print(np.percentile(distr, 50))
