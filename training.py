import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gin_backbone import GINBackbone
from models.gin_fusion import GINFusionNet

from utils import *
import random
import argparse

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

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data.x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_idx', type=int, required=True)
parser.add_argument("--model_idx", type=int, default=0, help="0: GIN, 1: GAT, 2: GAT_GCN, 3: GCN")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--fusion", type=int, default=None, help="Use fusion: 0 = transformer, 1 = crossattention")
parser.add_argument("--history_file", type=str, default=None, help="Custom CSV filename for training history")
parser.add_argument("--batch_size", type=int, default=512, help="Training and test batch size")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
parser.add_argument("--log_interval", type=int, default=20, help="Logging interval")

args = parser.parse_args()

#  在 parser 完成之后再取 dataset_name
dataset_list = ['davis', 'kiba']
dataset_name = dataset_list[args.dataset_idx]


TRAIN_BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.batch_size
LR = args.lr
NUM_EPOCHS = args.epochs
PATIENCE = args.patience
LOG_INTERVAL = args.log_interval

dataset = ['davis', 'kiba'][args.dataset_idx]

model_classes = [GINConvNet, GATNet, GAT_GCN, GCNNet]
backbone_classes = [GINFusionNet, GATNet, GAT_GCN, GCNNet]  # 注意 GINFusionNet 来自 gin_fusion.py

model_st = model_classes[args.model_idx].__name__
dataset_st = dataset

cuda_name = f"cuda:{args.cuda}"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

print('Learning rate:', LR)
print('Epochs:', NUM_EPOCHS)
print('Fusion mode:', args.fusion)
print('Running model:', model_st, 'on dataset:', dataset)

train_path = f'data/processed/{dataset}_train.pt'
valid_path = f'data/processed/{dataset}_valid.pt'

if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
    print('Please run create_data.py to prepare data in PyTorch format!')
    sys.exit()

train_data = TestbedDataset(root='data', dataset=dataset + '_train')
valid_data = TestbedDataset(root='data', dataset=dataset + '_valid')

g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

esm_dim = 1280
fusion_dim = 256

if args.fusion is not None:
    Backbone = backbone_classes[args.model_idx]
    if args.fusion == 0:
        from models.fusion_transformer import GNNTransformerFusionNet
        model = GNNTransformerFusionNet(Backbone(), esm_dim=esm_dim, fusion_dim=fusion_dim).to(device)
        fusion_tag = "transformer"
        print("Fusion mode: Transformer")
    elif args.fusion == 1:
        from models.fusion_crossattention import GNNCrossAttentionFusionNet
        model = GNNCrossAttentionFusionNet(Backbone(), esm_dim=esm_dim, fusion_dim=fusion_dim).to(device)
        fusion_tag = "crossattention"
        print("Fusion mode: CrossAttention")
    else:
        raise ValueError("Unsupported fusion type.")
else:
    model = model_classes[args.model_idx]().to(device)
    fusion_tag = "nofusion"
    print("Fusion mode: OFF (pure GNN)")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_mse = float('inf')
best_epoch = -1
counter = 0

model_file_name = f'model_{model_st}_{dataset}_{fusion_tag}.model'
result_file_name = f'result_{model_st}_{dataset}_{fusion_tag}.csv'
history = []

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)
    G, P = predicting(model, device, valid_loader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    history.append([epoch + 1] + ret)

    if ret[1] < best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str, ret)))
        best_mse = ret[1]
        best_epoch = epoch + 1
        counter = 0
        print(f'rmse improved at epoch {best_epoch}; best_mse={best_mse}')
    else:
        counter += 1
        print(f'No improvement since epoch {best_epoch} ({counter}/{PATIENCE})')
        if counter >= PATIENCE:
            print('Early stopping!')
            break

history_path = args.history_file if args.history_file else f"history_{model_st}_{dataset_st}_{fusion_tag}.csv"
pd.DataFrame(history, columns=['epoch', 'rmse', 'mse', 'pearson', 'spearman', 'ci']).to_csv(history_path, index=False)
print(f'Training history saved to {history_path}')
