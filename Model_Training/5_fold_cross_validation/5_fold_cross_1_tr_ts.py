#########################################################################
'''
Author:        Dipayan <dipayansarkar26@gmail.com>
Licence:       MIT (see LICENCE file)
Description:   This script used combined Arabidopsis data for 5 fold cross-validation.
'''
#########################################################################


import os, time, math, pickle, warnings, random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, balanced_accuracy_score,
    average_precision_score, roc_auc_score, auc, precision_recall_curve
)
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve 
from sklearn import metrics
# ──────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
EMB_PATH   = r"ARACoFusion-PPI/Embedding_Generation/Generated_embeddings/esm1b_C1_C2_C3.pkl"

CSV_DATA   = r"ARACoFusion-PPI/Datasets/Processed/Arabidopsis/concatenated_shuffled_c1c2c3.csv"

N_SPLITS   = 5                 
INNER_VAL  = 0.10        
BATCH      = 128
EPOCHS     = 40
LR         = 6.8138e-4
DROP       = 0.11848
GAMMA      = 3.89257
SMOOTH     = 0.05145
UNCERT_W   = 0.57906
HEADS      = 8
MC_PASSES  = 3
PATIENCE   = 10

CSV_OUT    = r"ARACoFusion-PPI/Model_Training/5_fold_cross_validation/seqcofusion_5fold_cv_metrics.csv"
FIG_DIR = r"ARACoFusion-PPI/Model_Training/5_fold_cross_validation/figures"
# ──────────────────────────────────────────────────────────────
def load_csv_pairs(csv_file: str):
    """Read the concatenated, shuffled C1 + C2 + C3 file and return
       (pairs, labels).  Sequences are upper-cased; labels coerced to int."""
    df = pd.read_csv(csv_file, dtype=str)
    df= df[:5000]
    df.columns = df.columns.str.strip().str.lower()

    df["col1"] = df["col1"].str.upper().str.strip()
    df["col2"] = df["col2"].str.upper().str.strip()

    df["interaction"] = pd.to_numeric(df["interaction"], errors="coerce")
    df = df.dropna(subset=["interaction"])
    df["interaction"] = df["interaction"].astype(int)

    return list(zip(df["col1"], df["col2"])), df["interaction"].tolist()


def build_tensor_dataset(pairs, labels, emb_dict):
    """Keep only pairs whose sequences are present in the
       pre-computed ESM1b embedding dictionary."""
    tensors, ys = [], []
    for (s1, s2), y in zip(pairs, labels):
        if s1 in emb_dict and s2 in emb_dict:
            tensors.append(torch.stack([emb_dict[s1], emb_dict[s2]], 0))
            ys.append(y)
    return torch.stack(tensors), torch.tensor(ys, dtype=torch.float32)

# ──────────────────────────────────────────────────────────────
def focal_loss_ls(logits, targets, gamma=GAMMA, smooth=SMOOTH):
    """Binary focal loss with label smoothing (Guo et al., ICML-2017)."""
    t = targets * (1 - smooth) + 0.5 * smooth
    p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    ce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    fl = ((1 - p) ** gamma * t + p ** gamma * (1 - t)) * ce
    return fl.mean()

# ──────────────────────────────────────────────────────────────
class SeqCoFusionBlock(nn.Module):
    def __init__(self, dim=1280, heads=HEADS):
        super().__init__()
        self.q1, self.kv1 = nn.Linear(dim, dim), nn.Linear(dim, dim * 2)
        self.q2, self.kv2 = nn.Linear(dim, dim), nn.Linear(dim, dim * 2)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=DROP,
                                          batch_first=True)

    def forward(self, p1, p2):                 # p1,p2 : (B,1,1280)
        Z1, _ = self.attn(self.q1(p1), *self.kv2(p2).chunk(2, -1))
        Z2, _ = self.attn(self.q2(p2), *self.kv1(p1).chunk(2, -1))
        return Z1.squeeze(1), Z2.squeeze(1)

class SeqCoFusionPPI(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.cross = SeqCoFusionBlock(dim)
        self.adapt = nn.Sequential(
            nn.Linear(dim, 768), nn.GELU(), nn.Dropout(DROP),
            nn.Linear(768, 512)
        )
        fusion_in = dim * 6 + 512 * 2         
        self.head = nn.Sequential(
            nn.BatchNorm1d(fusion_in),
            nn.Linear(fusion_in, 512), nn.GELU(), nn.Dropout(DROP),
            nn.Linear(512, 128),       nn.GELU(), nn.Dropout(DROP),
            nn.Linear(128, 1)
        )

    def forward(self, x):                     
        p1, p2 = x[:, 0, :], x[:, 1, :]
        c1, c2 = self.cross(p1.unsqueeze(1), p2.unsqueeze(1))
        prod   = p1 * p2
        diff   = (p1 - p2).abs()
        a1, a2 = self.adapt(p1), self.adapt(p2)
        z = torch.cat([p1, p2, c1, c2, prod, diff, a1, a2], -1)
        return self.head(z).squeeze(1)         # (B,) logits
# ──────────────────────────────────────────────────────────────
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temp

def tune_temperature(logits, labels):
    """1-parameter post-hoc calibration (Guo et al.)."""
    scaler = TempScaler().to(DEVICE)
    opt = torch.optim.LBFGS([scaler.temp], lr=0.1, max_iter=50)

    y = labels.float().to(DEVICE)

    def _closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits), y)
        loss.backward()
        return loss

    opt.step(_closure)
    return scaler.temp.detach().item()


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve 
from sklearn import metrics

def binary_metrics(y_true, y_prob, thr=0.5,evalset=''):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    
    ######### plot conf matrix #########
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix (Threshold = {thr:.2f})")
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix({evalset})_thr_{thr:.2f}.png"))
    plt.close()
    ######### plot ROC curve #########
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='green', label='AUROC score = {:.2f}'.format(roc_auc))
    plt.fill_between(fpr, tpr, alpha=0.1, color='green')######
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f"ROC Curve ({evalset})", loc='left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, f"roc_curve({evalset})_thr_{thr:.2f}.png"))
    plt.close()
    
    ########### plot PR curve #########
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_score = metrics.auc(recall, precision)
    # auprc_score
    plt.figure()
    plt.plot(recall, precision, color='blue', label='AUPRC-{:.2f}'.format(auprc_score))
    # pr_auc = metrics.auc(recall, precision)
    plt.fill_between(recall, precision, alpha=0.1, color='blue')######
    plt.title(f"Precision-Recall Curve ({evalset})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, f"pr_curve({evalset})_thr_{thr:.2f}.png"))
    plt.close()
    ######### end of plots #########
    spec = tn/(tn+fp) if (tn+fp) else 0
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_score = metrics.auc(recall, precision)
    sen  = recall_score(y_true, y_pred, zero_division=0) 
    bacc = balanced_accuracy_score(y_true, y_pred)
    npv  = tn / (tn + fn) if (tn + fn) else 0.0           # NPV
    
    return dict(
        ACC   = accuracy_score(y_true,y_pred),
        SEN   = sen,
        spec  = spec,
        prec  = precision_score(y_true,y_pred,zero_division=0),
        RECALL   = recall_score(y_true,y_pred,zero_division=0),
        F1    = f1_score(y_true,y_pred,zero_division=0),
        mcc   = matthews_corrcoef(y_true,y_pred),
        NPV   = npv,
        AUPR  = average_precision_score(y_true,y_prob),
        AUROC = roc_auc_score(y_true, y_prob),
        BACC  = bacc,
        AUPRC = auprc_score,
        
    )

# ──────────────────────────────────────────────────────────────
def run_fold(X, y, tr_idx, te_idx,fold_no = None):

    y_np = y.cpu().numpy()
    tr_idx, val_idx = train_test_split(
        tr_idx, test_size=INNER_VAL, stratify=y_np[tr_idx], random_state=0
    )

    tr_loader = Data.DataLoader(
        Data.TensorDataset(X[tr_idx], y[tr_idx]),
        batch_size=BATCH, shuffle=True
    )
    val_loader = Data.DataLoader(
        Data.TensorDataset(X[val_idx], y[val_idx]),
        batch_size=BATCH
    )
    te_loader = Data.DataLoader(
        Data.TensorDataset(X[te_idx], y[te_idx]),
        batch_size=BATCH
    )

    model = SeqCoFusionPPI().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = ReduceLROnPlateau(opt, patience=2, factor=0.5, verbose=False)

    best_val, best_state, patience_cnt = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); epoch_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits_mc = torch.stack([model(xb) for _ in range(MC_PASSES)], 0)
            probs_mc  = torch.sigmoid(logits_mc)
            loss_cls  = focal_loss_ls(logits_mc.mean(0), yb)
            loss_var  = probs_mc.var(0).mean()
            loss      = loss_cls + UNCERT_W * loss_var
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * len(xb)

        model.eval(); val_loss, logits_val, y_val = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="sum"
                ).item()
                logits_val.append(logits);  y_val.append(yb)
        val_loss /= len(val_idx)
        sched.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val, patience_cnt = val_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    logits_val = torch.cat(logits_val)
    y_val      = torch.cat(y_val)
    T = tune_temperature(logits_val, y_val)

    probs_val = torch.sigmoid(logits_val / T).cpu().numpy()
    thr_grid  = np.linspace(0.3, 0.9, 61)
    mccs      = [matthews_corrcoef(y_val.cpu().numpy(),
                                   (probs_val >= t).astype(int))
                 for t in thr_grid]
    best_thr  = thr_grid[int(np.argmax(mccs))]

    probs_te, y_te = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            logits = model(xb.to(DEVICE)) / T
            probs_te.append(torch.sigmoid(logits).cpu())
            y_te.append(yb)
    probs_te = torch.cat(probs_te).numpy()
    y_te     = torch.cat(y_te).numpy()

    return binary_metrics(y_te, probs_te, best_thr,evalset = f'Test Fold__{fold_no}') 

# ──────────────────────────────────────────────────────────────
def main():
    print("\n Loading ESM1b embeddings …", flush=True)
    emb_dict = pickle.load(open(EMB_PATH, "rb"))

    pairs, labels = load_csv_pairs(CSV_DATA)
    indices = np.random.permutation(len(pairs))
    pairs  = [pairs[i]  for i in indices]
    labels = [labels[i] for i in indices]

    X, y = build_tensor_dataset(pairs, labels, emb_dict)
    y_np = y.numpy()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    all_metrics = defaultdict(list)

    for fold, (tr, te) in enumerate(cv.split(X, y_np), 1):
        print(f"\n─────────────────────────────────────────")
        print(f" Fold {fold}/{N_SPLITS}  (train = 4 folds | test = 1 fold)")
        print(f"─────────────────────────────────────────")
        metrics = run_fold(X, y, tr, te, fold_no=fold)
        for k, v in metrics.items(): all_metrics[k].append(v)
        print("  ".join(f"{k}:{v:6.4f}" for k, v in metrics.items()))

    # ─── Aggregate across folds & export CSV ─────────────────
    print("\n═════════════════════════════════════════")
    print(" 5-fold cross-validation summary")
    print("═════════════════════════════════════════")

    df = pd.DataFrame(all_metrics,
                      index=[f"Fold{i}" for i in range(1, N_SPLITS + 1)])
    mean = df.mean()
    std  = df.std(ddof=1)
    df.loc["Mean±SD"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean, std)]

    for col in df.columns:
        print(f"{col:<6}: {mean[col]:6.4f} ± {std[col]:.4f}")

    df.to_csv(CSV_OUT, index_label="Fold")
    print(f"\n Per-fold metrics written to {CSV_OUT}")

    print("\nDone.")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
