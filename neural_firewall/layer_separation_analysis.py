"""
NPS Layer Separation Analysis
Consumes the existing 36-layer XSTest held-out activations.
No model loading or extraction is required.

Outputs:
  layer_separation.json
  layer_separation.csv
  layer_separation.png
  layer_separation_groups.png
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, balanced_accuracy_score

DEFAULT_DATA="phase1_real/xstest/full_layer_trajectories/trajectories.npz"
DEFAULT_METADATA="phase1_real/xstest/full_layer_trajectories/metadata.json"

def fisher_ratio(X,y):
    a,b=X[y==0],X[y==1]
    mu0,mu1=a.mean(0),b.mean(0)
    v0,v1=a.var(0,ddof=1),b.var(0,ddof=1)
    return float(np.mean((mu1-mu0)**2)/(np.mean(v0+v1)+1e-12))

def centroid_metrics(X,y):
    a,b=X[y==0],X[y==1]
    m0,m1=a.mean(0),b.mean(0)
    dist=float(np.linalg.norm(m1-m0))
    cos=float(np.dot(m0,m1)/(np.linalg.norm(m0)*np.linalg.norm(m1)+1e-12))
    return dist,cos

def cv_metrics(X,y):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,random_state=42))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scoring={"accuracy":"accuracy","balanced_accuracy":make_scorer(balanced_accuracy_score),"roc_auc":"roc_auc"}
    r=cross_validate(clf,X,y,cv=cv,scoring=scoring,n_jobs=-1)
    return {k:float(r["test_"+k].mean()) for k in scoring} | {
        "cv_accuracy_std":float(r["test_accuracy"].std())
    }

def mean_norm(X):
    return float(np.linalg.norm(X,axis=1).mean()) if len(X) else float("nan")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default=DEFAULT_DATA)
    p.add_argument("--metadata",default=DEFAULT_METADATA)
    p.add_argument("--out",default="phase1_real/xstest/layer_separation")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    z=np.load(args.data)
    A=z["activations"].astype(np.float32); y=z["labels"].astype(np.int64)
    with open(args.metadata,encoding="utf-8") as f: meta=json.load(f)
    ex=meta["examples"]
    pred=np.array([int(x["predicted"]) for x in ex])
    n,L,d=A.shape
    print("# NPS LAYER SEPARATION ANALYSIS")
    print(f"examples={n} layers={L} hidden={d}")
    print(f"safe={np.sum(y==0)} unsafe_detected={np.sum((y==1)&(pred==1))} unsafe_missed={np.sum((y==1)&(pred==0))}")
    rows=[]
    for l in range(L):
        X=A[:,l,:]
        dist,cos=centroid_metrics(X,y)
        cv=cv_metrics(X,y)
        det=X[(y==1)&(pred==1)]; miss=X[(y==1)&(pred==0)]
        gap=float(np.linalg.norm(det.mean(0)-miss.mean(0))) if len(det) and len(miss) else float("nan")
        r={"layer":l,"centroid_distance_safe_unsafe":dist,
           "centroid_cosine_safe_unsafe":cos,
           "fisher_ratio_safe_unsafe":fisher_ratio(X,y),
           "cv_accuracy_mean":cv["accuracy"],
           "cv_accuracy_std":cv["cv_accuracy_std"],
           "cv_balanced_accuracy_mean":cv["balanced_accuracy"],
           "cv_roc_auc_mean":cv["roc_auc"],
           "detected_vs_missed_centroid_distance":gap,
           "safe_mean_norm":mean_norm(X[y==0]),
           "unsafe_detected_mean_norm":mean_norm(det),
           "unsafe_missed_mean_norm":mean_norm(miss)}
        rows.append(r)
        print(f"L{l:02d} | AUC={r['cv_roc_auc_mean']:.4f} | BA={r['cv_balanced_accuracy_mean']:.4f} | Fisher={r['fisher_ratio_safe_unsafe']:.6g} | dist={dist:.4f} | FN-gap={gap:.4f}")
    best_auc=max(rows,key=lambda r:r["cv_roc_auc_mean"])
    best_f=max(rows,key=lambda r:r["fisher_ratio_safe_unsafe"])
    best_d=max(rows,key=lambda r:r["centroid_distance_safe_unsafe"])
    valid=[r for r in rows if np.isfinite(r["detected_vs_missed_centroid_distance"])]
    best_gap=max(valid,key=lambda r:r["detected_vs_missed_centroid_distance"])
    summary={"n_examples":n,"n_layers":L,"hidden_size":d,
             "note":"CV metrics are diagnostic only; this uses the held-out XSTest set and is not an independent generalization estimate.",
             "best_layer_by_cv_roc_auc":best_auc,
             "best_layer_by_fisher_ratio":best_f,
             "best_layer_by_centroid_distance":best_d,
             "largest_detected_vs_missed_gap":best_gap,"rows":rows}
    (out/"layer_separation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"layer_separation.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    layers=np.arange(L)
    auc=np.array([r["cv_roc_auc_mean"] for r in rows])
    dist=np.array([r["centroid_distance_safe_unsafe"] for r in rows])
    fig,ax=plt.subplots(figsize=(13,7)); ax.plot(layers,auc,"o-",label="5-fold linear-probe ROC-AUC"); ax.set_xlabel("Decoder layer"); ax.set_ylabel("ROC-AUC"); ax.set_ylim(.45,1.01); ax.grid(alpha=.25)
    ax2=ax.twinx(); ax2.plot(layers,dist,"s-",label="Safe/unsafe centroid distance"); ax2.set_ylabel("Centroid distance")
    ax.set_title("Layer-by-layer safe vs unsafe representation separation")
    a1,b1=ax.get_legend_handles_labels(); a2,b2=ax2.get_legend_handles_labels(); ax.legend(a1+a2,b1+b2)
    fig.tight_layout(); fig.savefig(out/"layer_separation.png",dpi=180); plt.close(fig)
    gap=np.array([r["detected_vs_missed_centroid_distance"] for r in rows])
    fig,ax=plt.subplots(figsize=(13,7)); ax.plot(layers,gap,"o-",label="Detected vs missed unsafe centroid distance"); ax.set_xlabel("Decoder layer"); ax.set_ylabel("Centroid distance"); ax.set_title("Do missed unsafe prompts follow a different representation?"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"layer_separation_groups.png",dpi=180); plt.close(fig)
    print("\n# KEY RESULTS")
    print(f"Best ROC-AUC: L{best_auc['layer']} ({best_auc['cv_roc_auc_mean']:.4f})")
    print(f"Best Fisher:  L{best_f['layer']} ({best_f['fisher_ratio_safe_unsafe']:.6g})")
    print(f"Best centroid distance: L{best_d['layer']} ({best_d['centroid_distance_safe_unsafe']:.4f})")
    print(f"Largest detected/missed gap: L{best_gap['layer']} ({best_gap['detected_vs_missed_centroid_distance']:.4f})")
    print(f"Saved to {out.resolve()}")

if __name__=="__main__": main()
