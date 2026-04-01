import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, confusion_matrix)
from tqdm import tqdm
import warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

sys.path.append('..')

from tools.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES, smiles_to_graph
from tools.model_config import config_dict
from models.kamt import KaMT


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss


class DynamicWeightLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            weighted_losses.append(torch.exp(-self.params[i]) * loss + self.params[i])
        return sum(weighted_losses), torch.exp(-self.params).detach().cpu().numpy()


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='in_proj'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)

    def restore(self, emb_name='in_proj'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    return (p_loss.sum() + q_loss.sum()) / 2


class AntibioticsDataset(Dataset):
    def __init__(self, csv_path, desc_path, config):
        self.df = pd.read_csv(csv_path)
        self.md = np.load(desc_path)['md']
        self.vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smi = self.df.iloc[idx]['smiles']
        label = self.df.iloc[idx]['label']
        md_feat = torch.tensor(self.md[idx], dtype=torch.float32)
        g = smiles_to_graph(smi, self.vocab, max_length=self.config['path_length'], n_virtual_nodes=2)
        return g, md_feat, torch.tensor([label], dtype=torch.float32)


def collate_fn(batch):
    graphs, mds, labels = zip(*batch)
    bg = dgl.batch(graphs)
    mds = torch.stack(mds, dim=0)
    labels = torch.stack(labels, dim=0)
    return bg, mds, labels


class KAMTForClassification(nn.Module):
    def __init__(self, pretrained_model, d_g_feats, num_labels=1):
        super().__init__()
        self.encoder = pretrained_model.model
        self.node_emb = pretrained_model.node_emb
        self.edge_emb = pretrained_model.edge_emb
        self.triplet_emb = pretrained_model.triplet_emb
        self.classifier = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_g_feats // 2, num_labels)
        )

    def forward(self, g, md):
        indicators = g.ndata['vavn']
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        batch_size = g.batch_size
        dummy_fp = torch.zeros((batch_size, 512)).to(node_h.device)

        if md.size(-1) != 201:
            md = md[:, :201]

        triplet_h = self.triplet_emb(node_h, edge_h, dummy_fp, md, indicators)
        triplet_h = self.encoder(g, triplet_h)
        graph_repr = triplet_h[indicators == 2]
        return self.classifier(graph_repr)


def train():
    parser = argparse.ArgumentParser(description="KAMT Antibiotics Finetune")
    parser.add_argument("--split", type=str, default="scaffold", choices=['random', 'scaffold'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--pretrained_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config_dict['base']
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    config['n_main_tasks'] = vocab.vocab_size

    data_root = f"../dataset/antibiotics/{args.split}"
    train_ds = AntibioticsDataset(f"{data_root}/train_aligned.csv", f"{data_root}/train_desc.npz", config)
    val_ds = AntibioticsDataset(f"{data_root}/val_aligned.csv", f"{data_root}/val_desc.npz", config)

    train_ds.md = np.nan_to_num(train_ds.md)
    val_ds.md = np.nan_to_num(val_ds.md)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    base_model = KaMT(config)
    print(f"Loading weights: {args.pretrained_path}")
    state_dict = torch.load(args.pretrained_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict, strict=False)

    model = KAMTForClassification(base_model, config['d_g_feats']).to(device)
    dwl = DynamicWeightLoss(num_tasks=2).to(device)
    focal_criterion = FocalLoss()
    fgm = FGM(model)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr * 0.2},
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': dwl.parameters(), 'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0
    best_cm = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, disable=(not sys.stdout.isatty()))

        for bg, md, labels in pbar:
            bg, md, labels = bg.to(device), md.to(device), labels.to(device)

            logits1, logits2 = model(bg, md), model(bg, md)
            l_focal = (focal_criterion(logits1, labels).mean() + focal_criterion(logits2, labels).mean()) / 2
            l_rd = compute_kl_loss(logits1, logits2)

            loss, _ = dwl(l_focal, l_rd * 0.1)
            optimizer.zero_grad()
            loss.backward()

            fgm.attack(epsilon=args.epsilon)
            loss_adv = focal_criterion(model(bg, md), labels).mean()
            loss_adv.backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for bg, md, labels in val_loader:
                bg, md = bg.to(device), md.to(device)
                scores = torch.sigmoid(model(bg, md))
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(scores.cpu().numpy())

        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()
        y_pred = (y_scores > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)

        scheduler.step()

        if auc > best_auc:
            best_auc = auc
            best_cm = confusion_matrix(y_true, y_pred)
            torch.save(model.state_dict(), f"../dataset/antibiotics/{args.split}/finetune_kamt_{args.split}_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"📊 Ep {epoch} | Loss: {total_loss / len(train_loader):.4f} | ROC: {auc:.4f} | PR: {pr_auc:.4f} | Best ROC: {best_auc:.4f}")

        if patience_counter >= args.patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

    print("\n" + "=" * 30)
    print(f"Pre-training completed! Best ROC-AUC: {best_auc:.4f}")
    if best_cm is not None:
        print("\n[Best Model Confusion Matrix - Validation Set]")
        print(f"True Negative (0): {best_cm[0, 0]:<5} | False Positive (0->1): {best_cm[0, 1]}")
        print(f"False Negative (1->0): {best_cm[1, 0]:<5} | True Positive (1): {best_cm[1, 1]}")
        print("-" * 30)
        # 打印可视化文本格式
        print(f"Predict\t 0 \t 1")
        print(f"True 0\t {best_cm[0, 0]} \t {best_cm[0, 1]}")
        print(f"True 1\t {best_cm[1, 0]} \t {best_cm[1, 1]}")
    print("=" * 30)


if __name__ == "__main__":
    train()
