# Updated create_data.py script with ESM embeddings and corrected dataset saving

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from utils import TestbedDataset
from transformers import EsmTokenizer, EsmModel
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load pretrained ESM model and tokenizer
esm_model_name = "facebook/esm1b_t33_650M_UR50S"
esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
esm_model = EsmModel.from_pretrained(esm_model_name)
esm_model.eval()
esm_model = esm_model.cuda() if torch.cuda.is_available() else esm_model

def get_esm_embedding(sequence):
    inputs = esm_tokenizer(sequence, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(esm_model.device)
    with torch.no_grad():
        outputs = esm_model(input_ids=input_ids)
    emb = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze().cpu().numpy()
    return emb.astype(np.float32)

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li',
        'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))) +
        [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

# Main pipeline
datasets = ['kiba', 'davis']
all_prots = []

for dataset in datasets:
    print(f'convert data from DeepDTA for  {dataset}')
    fpath = f'data/{dataset}/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) for d in ligands.keys()]
    prots = list(proteins.values())
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    split_opts = ['train', 'valid', 'test']
    folds = {'train': train_fold, 'valid': valid_fold, 'test': test_fold}

    for opt in split_opts:
        rows, cols = np.where(~np.isnan(affinity))
        rows, cols = rows[folds[opt]], cols[folds[opt]]
        with open(f'data/{dataset}_{opt}.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for i in range(len(rows)):
                ls = [drugs[rows[i]], prots[cols[i]], affinity[rows[i], cols[i]]]
                f.write(','.join(map(str, ls)) + '\n')
    all_prots += list(set(prots))
    print(f'dataset: {dataset}, train_fold: {len(train_fold)}, valid_fold: {len(valid_fold)}')

print("Generating ESM embeddings...")
esm_cache = {}
for prot in set(all_prots):
    esm_cache[prot] = get_esm_embedding(prot)
np.save('data/esm_cache.npy', esm_cache)
print('Saved ESM cache.')

# Build graph dict
compound_iso_smiles = set()
for dataset in datasets:
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(f'data/{dataset}_{split}.csv')
        compound_iso_smiles |= set(df['compound_iso_smiles'])

smile_graph = {smile: smile_to_graph(smile) for smile in compound_iso_smiles}

# Process and save datasets
for dataset in datasets:
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(f'data/{dataset}_{split}.csv')
        drugs = list(df['compound_iso_smiles'])
        prots = list(df['target_sequence'])
        ys = list(df['affinity'])
        esm_embed = [esm_cache[p] for p in prots]

        print(f'Preparing data/processed/{dataset}_{split}.pt ...')
        dataset_obj = TestbedDataset(
            root='data',
            dataset=f'{dataset}_{split}',
            xd=np.asarray(drugs),
            xt=np.asarray(esm_embed),
            y=np.asarray(ys),
            smile_graph=smile_graph,
            pre_transform=None  # 禁用 PyG 默认 pre_transform 处理
        )
