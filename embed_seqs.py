import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Union

import torch
from IO import loadTaxonomyTree, readVocabulary

from Model.EncoderModels import DeepurifyModel
from utils import getNumberOfPhylum, splitLongContig, reverseContigRepNormNumpy, ConvertSeqToImageTensorMoreFeatures

def embedding_contigs(
    contigName2seq: Dict[str, str],
    device: str,
    taxo_tree_path = "./Deepurify-DB/PyObjs/gtdb_taxonomy_tree.pkl",
    taxo_vocab_path = "./Deepurify-DB/Vocabs/taxa_vocabulary.txt",
    model_weigth_path = "./Deepurify-DB/CheckPoint/GTDB-clu-last.pth",
    mer3_vocabulary_path = "./Deepurify-DB/Vocabs/3Mer_vocabulary.txt",
    mer4_vocabulary_path = ".//Deepurify-DB/Vocabs/4Mer_vocabulary.txt",
    batch_size=4,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    model_config=None,
):
    pid = str(os.getpid())
    name2seq = contigName2seq
    if model_config is None:
        model_config = {
            "min_model_len": 1000,
            "max_model_len": 8192,
            "inChannel": 108,
            "expand": 1.2,
            "IRB_num": 2,
            "head_num": 6,
            "d_model": 864,
            "num_GeqEncoder": 6,
            "num_lstm_layers": 5,
            "feature_dim": 1024,
        }
    taxo_tree = loadTaxonomyTree(taxo_tree_path)
    taxo_vocabulary = readVocabulary(taxo_vocab_path)
    mer3_vocabulary = readVocabulary(mer3_vocabulary_path)
    mer4_vocabulary = readVocabulary(mer4_vocabulary_path)
    spe2index = {}
    index = 0
    for name, _ in taxo_vocabulary.items():
        if "s__" == name[0:3]:
            spe2index[name] = index
            index += 1
    model = DeepurifyModel(
        max_model_len=model_config["max_model_len"],
        in_channels=model_config["inChannel"],
        taxo_dict_size=len(taxo_vocabulary),
        vocab_3Mer_size=len(mer3_vocabulary),
        vocab_4Mer_size=len(mer4_vocabulary),
        phylum_num=getNumberOfPhylum(taxo_tree),
        species_num=len(spe2index),
        head_num=model_config["head_num"],
        d_model=model_config["d_model"],
        num_GeqEncoder=model_config["num_GeqEncoder"],
        num_lstm_layer=model_config["num_lstm_layers"],
        IRB_layers=model_config["IRB_num"],
        expand=model_config["expand"],
        feature_dim=model_config["feature_dim"],
        drop_connect_ratio=0.0,
        dropout=0.0,
    )
    model.to(device)
    ########### IMPORT ##########
    state = torch.load(model_weigth_path, map_location=torch.device(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    # Split contig if longer than max_model_len,
    # Since we split the long contigs, than we need to reverse to the original
    name2seq = splitLongContig(name2seq, max_model_len=cutSeqLength, min_model_len=model_config["min_model_len"], overlappingRatio=overlapping_ratio)
    names = []
    visRepVectorList = []
    batchList = []
    nsL = len(name2seq)
    k = 0
    for i, (name, seq) in enumerate(name2seq.items()):
        with torch.no_grad():
            ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = ConvertSeqToImageTensorMoreFeatures(
                model_config["max_model_len"], seq, mer3_vocabulary, mer4_vocabulary)
            ori_rev_tensor = ori_rev_tensor.to(device)  # [C, L]
            feature_3Mer = feature_3Mer.to(device)  # [L]
            feature_3Mer_rev_com = feature_3Mer_rev_com.to(device)  # [L]
            feature_4Mer = feature_4Mer.to(device)  # [L]
            feature_4Mer_rev_com = feature_4Mer_rev_com.to(device)  # [L]
            catedTensror = model.annotatedConcatTensors(ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com)
        names.append(name)
        batchList.append(catedTensror)
        if len(batchList) % batch_size == 0:
            if k % 5 == 0:
                statusStr = "    " + "PROCESSER {}, {:.4}% complete. (Current / Total) --> ({} / {})".format(
                    pid, (i + 1.0) * 100.0 / nsL + 0.0, k + 1, nsL
                )
                cn = len(statusStr)
                if cn < 150:
                    statusStr = statusStr + "".join([" " for _ in range(150 - cn)])
                statusStr += "\r"
                sys.stderr.write("%s\r" % statusStr)
                sys.stderr.flush()
            k += 1
            stacked = torch.stack(batchList, dim=0).to(device)
            with torch.no_grad():
                brepVectors = model.visionRep(stacked)
                for repVector in brepVectors.detach().cpu():
                    visRepVectorList.append(repVector)
            batchList = []
    if len(batchList) != 0:
        stacked = torch.stack(batchList, dim=0).to(device)
        with torch.no_grad():
            brepVectors = model.visionRep(stacked)
            for repVector in brepVectors.detach().cpu():
                visRepVectorList.append(repVector)
    assert len(names) == len(visRepVectorList), "The length is not equal with each other."
    name2repV = {}
    for name, visRepV in zip(names, visRepVectorList):
        name2repV[name] = visRepV

    # Reverse to original
    return reverseContigRepNormNumpy(name2repV)