import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "icl"))
import tasks
from tokspans import label_spans, target_ids
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")


def test_batteries_load():
    T = json.load(open(os.path.join(HERE, "..", "batteries", "icl_tasks.json")))
    assert len(tasks.LEARN_TASKS(T)) > 0
    assert all(len(v[0]) >= 3 and len(v[1]) >= 3 for v in tasks.CONCEPTS.values())


def test_concept12_twins_align():
    items = tasks.fam_concept12(TOK, smoke=True)
    assert items
    for it in items:
        ids = it.ids.tolist()
        assert target_ids(TOK, ids, it.target)[0]
        for tw in it.twins.values():
            assert len(tw.ids) == len(ids)
            assert [(a, b) for a, b, _ in tw.spans] == [(a, b) for a, b, _ in it.spans]
