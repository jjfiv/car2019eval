#%%
import attr
from typing import Dict, List
import gzip
import argparse
import json
from collections import defaultdict
from typing import DefaultDict, Set, Optional, Tuple

#%%
@attr.s
class TrecRunEntry(object):
    query_id = attr.ib(type=str)
    para_id = attr.ib(type=str)
    score = attr.ib(type=float)
    rank = attr.ib(type=int)
    system = attr.ib(type=str)

    def to_output(self):
        return "{0} Q0 {1} {2} {3} {4}".format(
            self.query_id, self.para_id, self.rank, self.score, self.system
        )


@attr.s
class QueryOutput(object):
    squid = attr.ib(type=str)
    run_id = attr.ib(type=str)
    title = attr.ib(type=str)
    paragraphs = attr.ib(type=List[Dict])
    paragraph_origins = attr.ib(type=List[Dict])
    query_facets = attr.ib(type=Optional[List[Dict]])

    def to_section_trecrun(self) -> List[TrecRunEntry]:
        system = self.run_id
        output = []
        for po in self.paragraph_origins:
            rank = po.get("rank", 0)
            para_id = po["para_id"]
            score = po["rank_score"]
            query_id = po["section_path"]
            output.append(TrecRunEntry(query_id, para_id, score, rank, system))
        return output

    def to_page_trecrun(self) -> List[TrecRunEntry]:
        """
        Note, the scores here may be nonsense: if each heading was a different query, they are not comparable.
        Perform Reciprocal Rank Fusion: if a paragraph
        """
        system = self.run_id
        query_id = self.squid
        output = []
        ranks_for_doc: DefaultDict[str, List[float]] = defaultdict(list)
        for po in self.paragraph_origins:
            rank = po.get("rank", 1)
            para_id = po["para_id"]
            score = po["rank_score"]
            ranks_for_doc[para_id].append(1.0 / max(1, rank))
        for (doc, recip_ranks) in ranks_for_doc.items():
            score = sum(recip_ranks)
            output.append(TrecRunEntry(query_id, para_id, score, -1, system))
        sorted_by_score = sorted(output, key=lambda tre: tre.score, reverse=True)
        for (i, tre) in enumerate(sorted_by_score):
            tre.rank = i + 1
        return sorted_by_score

    def para_id_order(self) -> List[str]:
        return [para["para_id"] for para in self.paragraphs]

    def paragraph_transitions(self):
        paras = self.para_id_order()
        pairs = []
        for i in range(len(paras) - 1):
            pairs.append((paras[i], paras[i + 1]))
        return pairs


def get_paragraph_transitions(input: str) -> Dict[str, List[Tuple[str, str]]]:
    found = {}
    with gzip.open(input, "rt") as fp:
        for line in fp:
            query_dict = json.loads(line)
            if "query_facets" not in query_dict:
                query_dict["query_facets"] = []
            query = QueryOutput(**query_dict)
            found[query.squid] = query.paragraph_transitions()
    return found


def convert_to_section_trecrun(input: str, output: str):
    with open(output, "w") as out:
        with gzip.open(input, "rt") as fp:
            for line in fp:
                query_dict = json.loads(line)
                if "query_facets" not in query_dict:
                    query_dict["query_facets"] = []
                query = QueryOutput(**query_dict)
                seen_docids: DefaultDict[str, Set[str]] = defaultdict(set)
                for entry in query.to_section_trecrun():
                    while entry.para_id in seen_docids[entry.query_id]:
                        entry.para_id += "_dup"
                    seen_docids[entry.query_id].add(entry.para_id)
                    print(entry.to_output(), file=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str)
    parser.add_argument("OUTPUT", type=str)
    args = parser.parse_args()

    convert_to_section_trecrun(args.INPUT, args.OUTPUT)
