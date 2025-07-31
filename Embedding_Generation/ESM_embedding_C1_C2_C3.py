#########################################################################
'''
Author:        Dipayan <dipayansarkar26@gmail.com>
Licence:       MIT (see LICENCE file)
Description:   This script generated embeddings for Arabidopsis C1, C2, and C3 sequences.
'''
#########################################################################


import torch
import pandas as pd
import pickle
from esm import pretrained
from typing import List

################################################################################
MAX_ESM_LEN = 1022          
STRIDE       = 512          
BATCH_SIZE   = 8            
EMBED_LAYER  = 33           
################################################################################
model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()
################################################################################
def load_all_sequences(files):
    seq_set = set()
    for file in files:
        df = pd.read_csv(file)
        seq_set.update(df["col1"].astype(str).str.upper().tolist())
        seq_set.update(df["col2"].astype(str).str.upper().tolist())
    return sorted(seq_set)  
################################################################################
@torch.inference_mode()
def _embed_many(pairs: List[tuple]) -> torch.Tensor:
    _, _, toks = batch_converter(pairs)
    toks = toks.cuda()
    out  = model(toks, repr_layers=[EMBED_LAYER], return_contacts=False)
    reps = out["representations"][EMBED_LAYER] 
    outs = []
    for i, (_, s) in enumerate(pairs):
        outs.append(reps[i, 1:len(s)+1].mean(dim=0))  
    return torch.stack(outs)                         


################################################################################
def _embed_long_sliding(seq: str,
                        window: int = MAX_ESM_LEN,
                        stride: int = STRIDE) -> torch.Tensor:
    chunks = [seq[i:i + window] for i in range(0, len(seq), stride)]
    if len(chunks[-1]) > window:
        chunks[-1] = chunks[-1][:window]

    reps = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = [(f"w{i+j}", s) for j, s in enumerate(chunks[i:i+BATCH_SIZE])]
        reps.append(_embed_many(batch))
    reps = torch.cat(reps, dim=0)                    

    return reps.mean(dim=0).cpu()                  
################################################################################
def compute_and_save_embeddings(all_sequences,
                                outfile="esm1b_C1_C2_C3.pkl"):
    embedding_dict = {}
    short_pairs = [(f"seq{i}", s) for i, s in enumerate(all_sequences)
                   if len(s) <= MAX_ESM_LEN]
    for j in range(0, len(short_pairs), BATCH_SIZE):
        batch_pairs = short_pairs[j:j+BATCH_SIZE]
        reps        = _embed_many(batch_pairs)
        for (name, seq), rep in zip(batch_pairs, reps):
            embedding_dict[seq] = rep
        print(f"\rShort | {j+len(batch_pairs):>6}/{len(short_pairs)} done",
              end="")
    long_seqs = [s for s in all_sequences if len(s) > MAX_ESM_LEN]
    for k, seq in enumerate(long_seqs, 1):
        embedding_dict[seq] = _embed_long_sliding(seq)
        print(f"\rLong  | {k:>6}/{len(long_seqs)} done", end="")
    with open(outfile, "wb") as f:
        pickle.dump(embedding_dict, f)
    print(f"\nSaved embeddings for {len(embedding_dict)} proteins → {outfile}")
################################################################################
if __name__ == "__main__":
    c1 = "/Datasets/Processed/Arabidopsis/c1filtered.csv"
    c2 = "/Datasets/Processed/Arabidopsis/c2filtered.csv"
    c3 = "/Datasets/Processed/Arabidopsis/c3filtered.csv"

    seqs = load_all_sequences([c1, c2, c3])
    compute_and_save_embeddings(seqs)
