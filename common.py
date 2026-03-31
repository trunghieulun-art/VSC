from typing import Dict, List, TypedDict


class ModelData(TypedDict):
    vocab: List[str]
    unigrams: Dict[str, int]
    bigrams: Dict[str, int]
