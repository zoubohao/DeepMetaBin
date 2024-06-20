import os
from typing import Dict, List, Tuple
import torch
import numpy as np


def getNumberOfPhylum(taxoTree: Dict) -> int:
    return len(taxoTree["Children"])


# Padding char "X"
nt2index = {"X": 0, "N": 1, "A": 2, "T": 3, "C": 4, "G": 5, "R": 1, "Y": 1, "M": 1, "K": 1, "W": 1, "H": 1, "B": 1, "V": 1, "S": 1, "D": 1}
nt2nt = {"A": "T", "T": "A", "C": "G", "G": "C"}


def buildSeqFeatures(seq: str, vocab_3Mer: Dict, vocab_4Mer: Dict) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    reverse_complement = []
    feature_3Mer = []
    feature_4Mer = []
    feature_3Mer_rev_com = []
    feature_4Mer_rev_com = []
    seqLen = len(seq)
    for i in range(seqLen):
        revIndex = seqLen - 1 - i
        if i + 3 <= seqLen:
            mer3 = seq[i: i + 3]
            if mer3 in vocab_3Mer:
                feature_3Mer.append(vocab_3Mer[mer3])
            else:
                feature_3Mer.append(vocab_3Mer["[UNK]"])
        if i + 4 <= seqLen:
            mer4 = seq[i: i + 4]
            if mer4 in vocab_4Mer:
                feature_4Mer.append(vocab_4Mer[mer4])
            else:
                feature_4Mer.append(vocab_4Mer["[UNK]"])
        if seq[revIndex] in nt2nt:
            reverse_complement.append(nt2nt[seq[revIndex]])
        else:
            reverse_complement.append("N")
        if i >= 3:
            rev_mer3 = "".join(reverse_complement[i - 3: i])
            if rev_mer3 in vocab_3Mer:
                feature_3Mer_rev_com.append(vocab_3Mer[rev_mer3])
            else:
                feature_3Mer_rev_com.append(vocab_3Mer["[UNK]"])
        if i >= 4:
            rev_mer4 = "".join(reverse_complement[i - 4: i])
            if rev_mer4 in vocab_4Mer:
                feature_4Mer_rev_com.append(vocab_4Mer[rev_mer4])
            else:
                feature_4Mer_rev_com.append(vocab_4Mer["[UNK]"])
    rev_mer3 = "".join(reverse_complement[i - 2: i + 1])
    if rev_mer3 in vocab_3Mer:
        feature_3Mer_rev_com.append(vocab_3Mer[rev_mer3])
    else:
        feature_3Mer_rev_com.append(vocab_3Mer["[UNK]"])
    rev_mer4 = "".join(reverse_complement[i - 3: i + 1])
    if rev_mer4 in vocab_4Mer:
        feature_4Mer_rev_com.append(vocab_4Mer[rev_mer4])
    else:
        feature_4Mer_rev_com.append(vocab_4Mer["[UNK]"])
    return reverse_complement, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com


def ConvertSeqToImageTensorMoreFeatures(max_model_len: int, seq: str, vocab_3Mer: Dict, vocab_4Mer: Dict) -> torch.Tensor:
    """
    This function requires the seq does not have padding char 'X'. The seq is the original seq.
    """
    # assert "X" not in seq, ValueError("'X' in the seq. ")
    seqLength = len(seq)
    assert seqLength <= max_model_len, "Your seq length is bigger than max_model_len."
    oriSeq = seq + "".join(["X" for _ in range(max_model_len - seqLength)])
    oriSeqIndex = torch.from_numpy(np.array(list(map(lambda x: nt2index[x], oriSeq)), dtype=np.int64)).view([max_model_len, 1])
    oriSeqTensor = torch.zeros([max_model_len, 6]).scatter_(dim=-1, index=oriSeqIndex, value=1.0).permute(1, 0).float()  # [6, max_model_len]
    # Other features
    reverse_complement, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = buildSeqFeatures(seq, vocab_3Mer, vocab_4Mer)
    reverse_complement = reverse_complement + ["X" for _ in range(max_model_len - seqLength)]
    rev_comp_index = torch.from_numpy(np.array(list(map(lambda x: nt2index[x], reverse_complement)), dtype=np.int64)).view([max_model_len, 1])
    rev_compTensor = torch.zeros([max_model_len, 6]).scatter_(dim=-1, index=rev_comp_index, value=1.0).permute(1, 0).float()  # [6, max_model_len]
    ###
    feature_3Mer += [0 for _ in range(max_model_len - len(feature_3Mer))]
    feature_3Mer = torch.from_numpy(np.array(feature_3Mer, dtype=np.int64))
    feature_4Mer += [0 for _ in range(max_model_len - len(feature_4Mer))]
    feature_4Mer = torch.from_numpy(np.array(feature_4Mer, dtype=np.int64))
    ###
    feature_3Mer_rev_com += [0 for _ in range(max_model_len - len(feature_3Mer_rev_com))]
    feature_3Mer_rev_com = torch.from_numpy(np.array(feature_3Mer_rev_com, dtype=np.int64))
    feature_4Mer_rev_com += [0 for _ in range(max_model_len - len(feature_4Mer_rev_com))]
    feature_4Mer_rev_com = torch.from_numpy(np.array(feature_4Mer_rev_com, dtype=np.int64))
    return torch.cat([oriSeqTensor, rev_compTensor], dim=0), feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com


def splitLongContig(
        name2seq: Dict[str, str],
        max_model_len: int,
        min_model_len: int,
        overlappingRatio=0.5):
    newName2seq = {}
    for name, seq in name2seq.items():
        seqLen = len(seq)
        if seqLen > max_model_len:
            start = 0
            k = 0
            while start + max_model_len <= seqLen:
                newName2seq[f"{name}___{str(k)}"] = seq[start: start + max_model_len]
                start += int(max_model_len * (1.0 - overlappingRatio))
                k += 1
            newName2seq[f"{name}___{str(k)}"] = seq[start:]
        else:
            newName2seq[name] = seq
    return newName2seq


def reverseContigRepNormNumpy(
    name2repV: Dict
):
    newName2repV = {}
    for name, repV in name2repV.items():
        assert len(repV.shape) == 1
        repV = repV.unsqueeze(0)
        visRepNorm = repV / repV.norm(dim=-1, keepdim=True)
        visRepNorm = visRepNorm.squeeze(0)
        assert len(visRepNorm.shape) == 1
        
        if "___" not in name:
            newName2repV[name] = visRepNorm
        else:
            contigName, _ = name.split("___")
            if contigName not in newName2repV:
                newName2repV[contigName] = [visRepNorm]
            else:
                newName2repV[contigName].append(visRepNorm)
    for name, repVlist in newName2repV.items():
        if isinstance(repVlist, list):
            meanRepV = torch.mean(torch.stack(repVlist, dim=0), dim=0, keepdim=True)
            meanRepV = meanRepV / meanRepV.norm(dim=-1, keepdim=True)
            newName2repV[name] = meanRepV.squeeze(0).numpy()
        else:
            newName2repV[name] = repVlist.numpy()
    return newName2repV