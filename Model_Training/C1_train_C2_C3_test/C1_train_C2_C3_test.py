#########################################################################
'''
Author:        Dipayan <dipayansarkar26@gmail.com>
Licence:       MIT (see LICENCE file)
Description:   This script used Arabidopsis C1 as training set and C2, C3 and Rice as test sets.
'''
#########################################################################


import os, time, pickle, math
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, balanced_accuracy_score,
    average_precision_score, roc_auc_score, auc
)
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
from sklearn.manifold import TSNE
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve 
from sklearn import metrics

CSV_OUT  = r"ARACoFusion-PPI/Model_Training/C1_train_C2_C3_test/c2_c3_c4_metrics22.csv"
FIG_DIR  = r"ARACoFusion-PPI/Model_Training/C1_train_C2_C3_test"

BATCH       = 128
EPOCHS      = 40
LR          = 0.0006813839376931293
DROP        = 0.11847993503770292
GAMMA       = 3.892573431254179         
SMOOTH      = 0.05145009657671993       
UNCERT_W    = 0.57905782932843          
HEADS       = 8           
MC_PASSES   = 3            
VAL_SPLIT   = 0.1
PATIENCE    = 16           

# SEED = 42


# THR_FIXED = 0.77 

# ---------- Metrics ----------
def compute_binary_metrics(y_true, y_prob, thr=0.5, evalset=""):
    y_pred = (y_prob>=thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
    
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
    sen  = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
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



def Tsne_RAW(X, y, data_name, transf_emb=False):
    """Projects embeddings X (N,2,1280) with labels y (N,1) to t-SNE."""
    if transf_emb:
        emb = X
    else:
        emb = X.flatten(start_dim=1, end_dim=2)  # (N, 2*1280)
        print(emb.shape)

    ############################# Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        metric='cosine'
    )
    embedding = tsne.fit_transform(emb)
    print(embedding.shape)
    print(y.shape)

    y_proj = y.flatten()
    print(y_proj.shape)
    mask_interacting     = (y_proj == 1)
    mask_non_interacting = (y_proj == 0)

    plt.scatter(
        embedding[mask_non_interacting, 0],
        embedding[mask_non_interacting, 1],
        s=5, label='Non-interacting', color='red'
    )
    plt.scatter(
        embedding[mask_interacting, 0],
        embedding[mask_interacting, 1],
        s=5, label='Interacting', color='green'
    )
    plt.xlabel('t-SNE dim. 1')
    plt.ylabel('t-SNE dim. 2')
    plt.title(f't-SNE projection of {data_name} embeddings')
    plt.legend(markerscale=2, fontsize='small', loc='upper right')
    plt.tight_layout()
    suffix = 'TRANSED' if transf_emb else 'RAW'
    plt.savefig(
        f'/C1_train_C2_C3_test/TSNE_{suffix}_{data_name}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()




def Tsne_project(
    X: torch.Tensor,
    y_true,
    y_probs,
    data_name: str,
    threshold: float = 0.5,
    transf_emb: bool = False
):


    if transf_emb:
        emb = X
    else:
        emb = X.flatten(start_dim=1, end_dim=2)

    if isinstance(emb, torch.Tensor):
        emb_np = emb.detach().cpu().numpy()
    else:
        emb_np = np.array(emb)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        metric='cosine'
    )
    emb_2d = tsne.fit_transform(emb_np) 
    def to_numpy_1d(x, dtype):
        """Helper: tensor → cpu numpy, or leave array unchanged."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(-1).astype(dtype)
        else:
            return np.array(x).reshape(-1).astype(dtype)

    y_true_np  = to_numpy_1d(y_true,  int)    
    y_probs_np = to_numpy_1d(y_probs, float) 


    y_pred_np = (y_probs_np >= threshold).astype(int)

    tp_mask = (y_pred_np == 1) & (y_true_np == 1)
    tn_mask = (y_pred_np == 0) & (y_true_np == 0)
    fp_mask = (y_pred_np == 1) & (y_true_np == 0)
    fn_mask = (y_pred_np == 0) & (y_true_np == 1)


    plt.scatter(emb_2d[tn_mask,0], emb_2d[tn_mask,1],
                s=5, label='True Negative (TN)', color='blue')
    plt.scatter(emb_2d[tp_mask,0], emb_2d[tp_mask,1],
                s=5, label='True Positive (TP)', color='green')
    plt.scatter(emb_2d[fp_mask,0], emb_2d[fp_mask,1],
                s=5, label='False Positive (FP)', color='orange')
    plt.scatter(emb_2d[fn_mask,0], emb_2d[fn_mask,1],
                s=5, label='False Negative (FN)', color='purple')

    plt.xlabel('t-SNE dim. 1')
    plt.ylabel('t-SNE dim. 2')
    plt.title(f't-SNE projection of {data_name} embeddings')
    plt.legend(markerscale=2, fontsize='small', loc='upper right')
    plt.tight_layout()

    suffix = 'TRANSF_TPFP' if transf_emb else 'RAW'
    plt.savefig(
        f'/C1_train_C2_C3_test/TSNE_{suffix}_{data_name}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()



def load_csv_pairs(csv_file):

    df = pd.read_csv(csv_file, dtype=str)      
    df.columns = df.columns.str.strip().str.lower()  
    df["col1"] = df["col1"].str.upper().str.strip()
    df["col2"] = df["col2"].str.upper().str.strip()
    df["interaction"] = pd.to_numeric(df["interaction"], errors="coerce")
    bad = df["interaction"].isna().sum()
    if bad:
        print(f"[load_csv_pairs] WARNING: dropped {bad} malformed rows in {csv_file}")
    df = df.dropna(subset=["interaction"])
    df["interaction"] = df["interaction"].astype(int)
    seq_pairs = list(zip(df["col1"], df["col2"]))
    labels    = df["interaction"].tolist()
    return seq_pairs, labels

def build_tensor_dataset(pairs, labels, emb_dict,data_name="", for_c4 = False):
    tensors, lab = [], []
    for (s1, s2), y in zip(pairs, labels):


        if for_c4:
            if s1 in emb_dict and s2 in emb_dict:
                emb1 = emb_dict[s1]
                emb2 = emb_dict[s2]
                emb1 = torch.as_tensor(emb1, dtype=torch.float32, device=DEVICE)
                emb2 = torch.as_tensor(emb2, dtype=torch.float32, device=DEVICE)
                pair_tensor = torch.stack([emb1, emb2], dim=0)  # (2, emb_dim)
                tensors.append(pair_tensor)
                lab.append(y)
                
        else:
            if s1 in emb_dict and s2 in emb_dict:
                tensors.append(torch.stack([emb_dict[s1], emb_dict[s2]], 0))
                lab.append(y)
                
    #########################################
    X = []
    missing = 0
    for a, b in pairs:
        try:
                emb_a = emb_dict[a].cpu().numpy()
                emb_b = emb_dict[b].cpu().numpy()
                X.append(np.stack([emb_a, emb_b], 0))
        except KeyError:
            missing += 1
    if missing:
        print(f"[warn] {missing} pairs skipped – embedding not found.")
        
    X = torch.tensor(np.stack(X, 0), dtype=torch.float32)
    y = torch.tensor(labels[:len(X)], dtype=torch.float32).unsqueeze(1)
    
    print("Printing shapes of X and y:")
    print(f"Data name: {data_name}")
    print(X.shape, y.shape)
    #########################################
    Tsne_RAW(X, y, data_name, transf_emb=False)  # save raw Tsne plot

    return torch.stack(tensors), torch.tensor(lab, dtype=torch.float)







# ---------- Losses ----------
def focal_loss_ls(logits, targets, gamma=GAMMA, smooth=SMOOTH):
    t = targets.float() * (1-smooth) + 0.5*smooth
    p = torch.sigmoid(logits).clamp(1e-5, 1-1e-5)
    ce = -(t*torch.log(p)+(1-t)*torch.log(1-p))
    fl = ((1-p)**gamma * t + p**gamma*(1-t)) * ce
    return fl.mean()




########################################################################################################
########################################################################################################
# ---------- Model ----------
class SeqCoFusionBlock(nn.Module):
    def __init__(self, dim=1280, heads=HEADS):
        super().__init__()
        self.q1 = nn.Linear(dim,dim)
        self.kv1 =  nn.Linear(dim,dim*2)
        self.q2 = nn.Linear(dim,dim)
        self.kv2 = nn.Linear(dim,dim*2)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=DROP,batch_first=True)
    def forward(self, p1, p2):                               # (B,1,1280)
        Z1,_ = self.attn(self.q1(p1), *self.kv2(p2).chunk(2,-1))
        Z2,_ = self.attn(self.q2(p2), *self.kv1(p1).chunk(2,-1))
        return Z1.squeeze(1), Z2.squeeze(1)

class SeqCoFusionPPI(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.cross = SeqCoFusionBlock(dim)
        self.adapt = nn.Sequential(
            nn.Linear(dim, 768), 
            nn.GELU(), 
            nn.Dropout(DROP),
            nn.Linear(768, 512))
        fusion_in = dim * 6 + 512 * 2         
        self.head_bn1   = nn.BatchNorm1d(fusion_in)
        self.head_fc1   = nn.Linear(fusion_in, 512)
        self.head_act1  = nn.GELU()
        self.head_drop1 = nn.Dropout(DROP)
        self.head_fc2   = nn.Linear(512, 128)
        self.head_act2  = nn.GELU()
        self.head_drop2 = nn.Dropout(DROP)

        self.head_fc3   = nn.Linear(128, 1)
    def forward(self,x):                      
        p1,p2 = x[:,0,:], x[:,1,:]
        c1,c2 = self.cross(p1.unsqueeze(1),p2.unsqueeze(1))
        prod  = p1*p2
        diff  = (p1-p2).abs()
        a1,a2 = self.adapt(p1), self.adapt(p2)
        z = torch.cat([p1,p2,c1,c2,prod,diff,a1,a2],-1)
        h = self.head_bn1(z)
        h = self.head_fc1(h)
        h = self.head_act1(h)
        h = self.head_drop1(h)
        h = self.head_fc2(h)
        h = self.head_act2(h)
        h = self.head_drop2(h)
        feats_head = h
        logits = self.head_fc3(feats_head).squeeze(1)
        return logits,feats_head    



########################################################################################################
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1)*1.0)
    def forward(self, logits):                 
        return logits / self.temp

def tune_temperature(logits, labels):
    scaler = TempScaler().to(DEVICE)
    opt = torch.optim.LBFGS([scaler.temp], lr=0.1, max_iter=50)
    y = labels.float().to(DEVICE)
    def _closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits), y)
        loss.backward(); return loss
    opt.step(_closure)
    return scaler.temp.detach().item()


def main():
    # ---- load embedding dict ----
    emb = pickle.load(open(r"ARACoFusion-PPI/Embedding_Generation/Generated_embeddings/esm1b_C1_C2_C3.pkl","rb"))
    emb2 = pickle.load(open(r"ARACoFusion-PPI/Embedding_Generation/Generated_embeddings/esm1b_rice.pkl","rb"))

    # ---- C1 training / validation ----
    pairs, labels = load_csv_pairs(r"ARACoFusion-PPI/Datasets/Processed/Arabidopsis/c1filtered.csv")
    X, y = build_tensor_dataset(pairs, labels, emb,data_name="C1")
    idx = np.arange(len(X)); np.random.shuffle(idx)
    split = int(len(X)*(1-VAL_SPLIT))
    tr_idx, val_idx = idx[:split], idx[split:]
    train_loader = Data.DataLoader(
        Data.TensorDataset(X[tr_idx], y[tr_idx]), batch_size=BATCH,
        shuffle=True)
    val_loader   = Data.DataLoader(
        Data.TensorDataset(X[val_idx], y[val_idx]), batch_size=BATCH)

    model = SeqCoFusionPPI().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = ReduceLROnPlateau(opt, patience=2, factor=0.5, verbose=True)

    best_val, patience_cnt = 1e9, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss=0
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits_mc = torch.stack([model(xb)[0] for _ in range(MC_PASSES)],0)
            probs_mc  = torch.sigmoid(logits_mc)
            loss_cls  = focal_loss_ls(logits_mc.mean(0), yb)
            loss_var  = probs_mc.var(0).mean()
            loss      = loss_cls + UNCERT_W*loss_var
            opt.zero_grad(); loss.backward()
            opt.step()
            epoch_loss += loss.item()*len(xb)
        # ---- validation loss ----
        model.eval(); 
        val_loss, logits_list, labels_list = 0,[],[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits,_ = model(xb)
                val_loss += F.binary_cross_entropy_with_logits(logits, yb).item()*len(xb)
                logits_list.append(logits); labels_list.append(yb)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)
        print(f"Epoch {epoch:02d} | train-loss {epoch_loss/len(train_loader.dataset):.4f}"
              f" | val-loss {val_loss:.4f}")
        if val_loss < best_val-1e-4: best_val = val_loss; patience_cnt=0
        else: patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping."); break

    # ---- temperature scaling on validation logits ----
    logits_val = torch.cat(logits_list)
    labels_val = torch.cat(labels_list)
    T = tune_temperature(logits_val, labels_val)
    print(f"Calibrated temperature T = {T:.3f}")
    # best threshold (max MCC) on validation set
    probs_val = torch.sigmoid(logits_val/T).cpu().numpy()
    yv = labels_val.cpu().numpy()
    
    thr_grid = np.linspace(0.3,0.9,61)
    mccs = [matthews_corrcoef(yv,(probs_val>=t).astype(int)) for t in thr_grid]
    best_thr = thr_grid[int(np.argmax(mccs))]
    # best_thr = THR_FIXED
    print(f"Optimal threshold (MCC) = {best_thr:.3f}")


    results = []
    ump1, ump2, ump3 = [], [], []
    
    # torch.save(model.state_dict(), r"model1.pth")
    # ---- evaluation on C2 / C3 ----
    for name,csv in [("C2",r"ARACoFusion-PPI/Datasets/Processed/Arabidopsis/c2filtered.csv")]:
        pairs, labels = load_csv_pairs(csv)
        Xtest, ytest  = build_tensor_dataset(pairs, labels, emb,data_name=name)
        model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(Xtest),BATCH):
                xb = Xtest[i:i+BATCH].to(DEVICE)
                logits,x1 = model(xb)
                logits = logits / T
                ump1.append((x1.cpu()/T))
                preds.append(torch.sigmoid(logits).cpu())
                
                
        probs = torch.cat(preds).numpy()
        mets  = compute_binary_metrics(np.array(labels)[:len(probs)], probs, best_thr, evalset=name)
        print(f"\n{name} results  (thr={best_thr:.3f})")
        for k,v in mets.items():
            print(f"{k:<6}: {v:.4f}")
            
        ump1_tensor = torch.cat(ump1, dim=0)
        ump1_np = ump1_tensor.numpy()
        results.append(dict(DATASET=name, **mets))
        ump1_np = ump1_tensor.numpy()
        Tsne_project(X=ump1_np, y_probs=probs,y_true=ytest, data_name=name, transf_emb=True)
        probs = None
        ytest = None



    # ---- evaluation on C2 / C3 ----
    for name,csv in [("C3",r"ARACoFusion-PPI/Datasets/Processed/Arabidopsis/c3filtered.csv")]:
        pairs, labels = load_csv_pairs(csv)
        Xtest, ytest  = build_tensor_dataset(pairs, labels, emb,data_name=name)
        model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(Xtest),BATCH):
                xb = Xtest[i:i+BATCH].to(DEVICE)
                logits,x1 = model(xb)
                logits = logits / T
                ump2.append((x1.cpu()/T))
                preds.append(torch.sigmoid(logits).cpu())
                
                
        probs = torch.cat(preds).numpy()
        mets  = compute_binary_metrics(np.array(labels)[:len(probs)], probs, best_thr, evalset=name)
        print(f"\n{name} results  (thr={best_thr:.3f})")
        for k,v in mets.items():
            print(f"{k:<6}: {v:.4f}")
            
        ump2_tensor = torch.cat(ump2, dim=0)
        ump2_np = ump2_tensor.numpy()
        results.append(dict(DATASET=name, **mets))
        ump2_np = ump2_tensor.numpy()
        Tsne_project(X=ump2_np, y_probs=probs,y_true=ytest, data_name=name, transf_emb=True)
        probs = None
        ytest = None



    # ---- evaluation on c4 ----
    for name,csv in [("C4",r"ARACoFusion-PPI/Datasets/Processed/Rice/Rice_PPI.csv")]:
        pairs, labels = load_csv_pairs(csv)
        Xtest, ytest  = build_tensor_dataset(pairs, labels, emb_dict=emb2, data_name=name, for_c4=True)
        model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(Xtest),BATCH):
                xb = Xtest[i:i+BATCH].to(DEVICE)
                logits,x2 = model(xb)
                logits = logits / T
                ump3.append((x2.cpu()/T))
                preds.append(torch.sigmoid(logits).cpu())
        probs = torch.cat(preds).numpy()
        mets  = compute_binary_metrics(np.array(labels)[:len(probs)], probs, best_thr, evalset=name)
        print(f"\n{name} results  (thr={best_thr:.3f})")
        for k,v in mets.items():
            print(f"{k:<6}: {v:.4f}")
        ump3_tensor = torch.cat(ump3, dim=0)
        results.append(dict(DATASET=name, **mets))
        ump3_np = ump3_tensor.numpy()
        Tsne_project(X=ump3_np, y_probs=probs,y_true=ytest, data_name=name, transf_emb=True)
        
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUT, index=False, float_format="%.4f")
    print("\n──────────── Summary (saved to CSV) ────────────")
    print(df.to_string(index=False))




if __name__ == "__main__":
    main()











