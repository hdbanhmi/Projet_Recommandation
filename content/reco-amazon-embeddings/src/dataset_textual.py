
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from cornac.datasets import amazon_clothing

UserId = str
ItemId = str
Rating = float
Triplet = Tuple[UserId, ItemId, Rating]

@dataclass
class TextualDataset:
    train: List[Triplet]
    test: List[Triplet]
    item_texts: Dict[ItemId, str]
    users: List[UserId]
    items: List[ItemId]

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def load_raw():
    fb = amazon_clothing.load_feedback()
    texts, ids = amazon_clothing.load_text()
    item_texts = {iid: clean_text(txt) for txt,iid in zip(texts,ids)}
    fb = [(u,i,float(r)) for u,i,r in fb if i in item_texts]
    return fb, item_texts

def leave_one_out(fb,seed=42):
    random.seed(seed)
    by_user = defaultdict(list)
    for u,i,r in fb:
        by_user[u].append((u,i,r))

    train,test=[],[]
    for u,trips in by_user.items():
        if len(trips)==1:
            train.extend(trips)
            continue
        idx=random.randrange(len(trips))
        for j,t in enumerate(trips):
            (test if j==idx else train).append(t)
    return train,test

def load_textual_dataset(seed=42):
    fb,item_texts=load_raw()
    train,test=leave_one_out(fb,seed)
    users=sorted({u for u,_,_ in fb})
    items=sorted(item_texts.keys())
    return TextualDataset(train=train,test=test,item_texts=item_texts,users=users,items=items)
