from typing import Dict, List, Tuple
import pickle

def readVocabulary(path: str) -> Dict:
    vocabulary = {}
    with open(path, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            vocabulary[oneLine[0]] = int(oneLine[1])
    return vocabulary


def loadTaxonomyTree(pkl_path: str) -> Dict:
    with open(pkl_path, mode="rb") as rb:
        tree = pickle.load(rb)
    return tree