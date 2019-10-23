#%%
import attr
from typing import Dict, List
import gzip
import argparse
import json

#%%
@attr.s
class SectionTrecRunEntry(object):
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
    query_facets = attr.ib(type=List[Dict])
    paragraphs = attr.ib(type=List[Dict])
    paragraph_origins = attr.ib(type=List[Dict])

    def to_section_trecrun(self) -> List[SectionTrecRunEntry]:
        system = self.run_id
        output = []
        for po in self.paragraph_origins:
            rank = po["rank"]
            para_id = po["para_id"]
            score = po["rank_score"]
            query_id = po["section_path"]
            output.append(SectionTrecRunEntry(query_id, para_id, score, rank, system))
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str)
    parser.add_argument("OUTPUT", type=str)
    args = parser.parse_args()

    with open(args.OUTPUT, "w") as out:
        with gzip.open(args.INPUT, "rt") as fp:
            for line in fp:
                query = QueryOutput(**json.loads(line))
                for entry in query.to_section_trecrun():
                    print(entry.to_output(), file=out)
