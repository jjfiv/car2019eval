#%%
import os
import glob
import ujson as json
import attr
from typing import Optional, Dict, List


#%%
@attr.s
class QueryPassageLabel(object):
    query_id = attr.ib(type=str)
    paragraph_id = attr.ib(type=str)
    header_id = attr.ib(type=Optional[str])
    relevance_level = attr.ib(type=str)

    def int_score(self) -> int:
        label = self.relevance_level
        if label == "NonRel":
            return 0
        elif label == "CanLabel":
            return 1
        elif label == "ShouldLabel":
            return 2
        else:
            assert label == "MustLabel"
            return 3


@attr.s
class TransitionLabel(object):
    query_id = attr.ib(type=str)
    para1_id = attr.ib(type=str)
    para2_id = attr.ib(type=str)
    label = attr.ib(type=str)

    def int_score(self) -> int:
        if self.label == "SwitchTransition":
            return 0
        elif self.label == "SameTransition":
            return 1
        else:
            assert self.label == "AppropriateTransition"
            return 2


#%%
@attr.s
class LabelSavedData(object):
    assessor_data = attr.ib()
    nonrelevant_state = attr.ib()
    facet_state = attr.ib()
    notes_state = attr.ib()
    nonrelevant_state2 = attr.ib()
    transition_label_state = attr.ib()

    def non_rel_labels(self) -> List[QueryPassageLabel]:
        output = []
        for [info, _timing] in self.nonrelevant_state2:
            query_id = info["query_id"]
            paragraph_id = info["paragraph_id"]
            output.append(
                QueryPassageLabel(
                    query_id, paragraph_id, header_id=None, relevance_level="NonRel"
                )
            )
        return output

    def rel_labels(self) -> List[QueryPassageLabel]:
        output = []
        for [info, value_list] in self.facet_state:
            query_id = info["query_id"]
            paragraph_id = info["paragraph_id"]
            for values in value_list:
                header_id = values["value"]["facet"]["heading_id"]
                relevance = values["value"]["relevance"]
                output.append(
                    QueryPassageLabel(query_id, paragraph_id, header_id, relevance)
                )
        return output

    def all_labels(self) -> List[QueryPassageLabel]:
        output = []
        output.extend(self.non_rel_labels())
        output.extend(self.rel_labels())
        return output

    def transition_labels(self) -> List[TransitionLabel]:
        output = []
        for [info, value_dict] in self.transition_label_state:
            query_id = info["query_id"]
            paragraph_id1 = info["paragraph_id1"]
            paragraph_id2 = info["paragraph_id2"]
            label = value_dict["value"]
            output.append(
                TransitionLabel(query_id, paragraph_id1, paragraph_id2, label)
            )
        return output


#%%
passage_labels = []
transition_labels = []

for assesment_file in glob.glob(
    "data/car-y3-assessments-processed/car-y3-manual-assessments/*.json"
):
    with open(assesment_file) as fp:
        dat = LabelSavedData(**json.load(fp))
        passage_labels.extend(dat.all_labels())
        transition_labels.extend(dat.transition_labels())

len(passage_labels), len(transition_labels)

#%%
from collections import Counter

by_query = Counter([p.query_id for p in passage_labels])
len(by_query), sorted(by_query.keys())

#%%
