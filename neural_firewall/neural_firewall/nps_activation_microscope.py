#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def args():
    p=argparse.ArgumentParser(description="NPS Neural Activation Microscope")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompts", help="text file, one prompt per line")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--artifacts", default=None)
    p.add_argument("--out", default="activation_microscope")
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    p.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model in 8-bit if bitsandbytes is available.",
    )
    return p.parse_args()

def load_prompts(a):
    if a.prompt is not None: return [a.prompt]
    return [x.strip() for x in Path(a.prompts).read_text(encoding="utf-8").splitlines() if x.strip()]

def load_probes(root):
    if not root: return {}
    root=Path(root); out={}
    for l in (19,20,21,22):
        w=root/f"unsafe_intent__layer{l}.weight.npy"
        m=root/f"unsafe_intent__layer{l}.meta.json"
        if w.exists() and m.exists():
            meta=json.loads(m.read_text(encoding="utf-8"))
            out[l]={"weight":np.load(w).astype(np.float32),
                    "bias":float(meta.get("bias",0.0)),
                    "threshold":float(meta["threshold"])}
    return out

def main():
    a=args()
    prompts=load_prompts(a)
    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available.")
        device = torch.device("cuda")
    elif a.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.dtype == "float32":
        dtype = torch.float32
    elif a.dtype == "float16":
        dtype = torch.float16
    elif a.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        # GTX 1650: float16 is the sensible default on CUDA.
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("[environment]")
    print("  torch:",torch.__version__)
    print("  CUDA available:",torch.cuda.is_available())
    print("  device:",device)
    if device.type=="cuda":
        print("  GPU:",torch.cuda.get_device_name(0))
        print("  VRAM:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),"GB")

    print("[model] loading",a.model)
    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    load_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if a.load_in_8bit:
        if device.type != "cuda":
            raise RuntimeError("--load-in-8bit requires CUDA.")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = {"": "cuda"}

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        **load_kwargs,
    )

    if not a.load_in_8bit:
        model = model.to(device)

    model.eval()
    layers=model.model.layers
    print("[model] decoder layers:",len(layers))
    print("[model] hidden size:",model.config.hidden_size)

    captured={}
    handles=[]
    for i,layer in enumerate(layers):
        def hook(module, inputs, i=i):
            captured[i]=inputs[0].detach().float().cpu()
        handles.append(layer.register_forward_pre_hook(hook))

    records=[]
    try:
        for n,prompt in enumerate(prompts):
            enc=tok(prompt,return_tensors="pt",truncation=True,max_length=a.max_input_tokens)
            ids=enc["input_ids"].to(device); mask=enc["attention_mask"].to(device)
            captured.clear()
            with torch.inference_mode(): model(input_ids=ids,attention_mask=mask)
            last=int(mask.sum().item())-1
            acts=np.stack([captured[i][0,last].numpy() for i in range(len(layers))]).astype(np.float32)
            records.append({"prompt":prompt,"tokens":int(mask.sum()),"activations":acts})
            print(f"[{n+1}/{len(prompts)}] captured {acts.shape}  {prompt[:80]!r}")
    finally:
        for h in handles: h.remove()

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"raw_activations.npz",**{f"prompt_{i}":r["activations"] for i,r in enumerate(records)})
    (out/"prompts.json").write_text(json.dumps([{"index":i,"prompt":r["prompt"],"tokens":r["tokens"]} for i,r in enumerate(records)],indent=2,ensure_ascii=False),encoding="utf-8")

    X=np.concatenate([r["activations"] for r in records])
    mean=X.mean(0); U,S,Vt=np.linalg.svd(X-mean,full_matrices=False)
    C=(X-mean)@Vt[:3].T
    coords={}; off=0
    for i,r in enumerate(records):
        n=r["activations"].shape[0]; coords[i]=C[off:off+n]; off+=n
    np.savez(out/"pca_projection.npz",mean=mean.astype(np.float32),components=Vt[:3].astype(np.float32),explained_variance_ratio=((S[:3]**2)/(S**2).sum()).astype(np.float32))
    np.savez(out/"trajectory_coordinates.npz",**{f"prompt_{i}":c.astype(np.float32) for i,c in coords.items()})

    import matplotlib.pyplot as plt
    # 2D PCA
    fig,ax=plt.subplots(figsize=(10,7))
    for i,r in enumerate(records):
        c=coords[i]; ax.plot(c[:,0],c[:,1],marker="o",markersize=3,label=r["prompt"][:45])
    ax.set(xlabel="PC1",ylabel="PC2",title="Qwen activation trajectory through decoder layers"); ax.grid(alpha=.25)
    if len(records)<=10: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"trajectory_2d.png",dpi=160); plt.close(fig)

    # 3D PCA
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d")
    for i,r in enumerate(records):
        c=coords[i]; ax.plot(c[:,0],c[:,1],c[:,2],marker="o",markersize=3,label=r["prompt"][:35])
    ax.set(xlabel="PC1",ylabel="PC2",zlabel="PC3",title="3D activation trajectories")
    if len(records)<=10: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"trajectory_3d.png",dpi=160); plt.close(fig)

    # Norm and movement
    fig,ax=plt.subplots(figsize=(10,6))
    for i,r in enumerate(records): ax.plot(np.linalg.norm(r["activations"],axis=1),marker="o",markersize=3,label=r["prompt"][:45])
    ax.set(xlabel="Decoder layer",ylabel="||h||2",title="Activation norm by layer"); ax.grid(alpha=.25)
    if len(records)<=10: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"activation_norm.png",dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,6))
    for i,r in enumerate(records):
        x=r["activations"]; ax.plot(np.linalg.norm(x[1:]-x[:-1],axis=1),marker="o",markersize=3,label=r["prompt"][:45])
    ax.set(xlabel="Transition to layer",ylabel="||h_l-h_(l-1)||2",title="Layer-to-layer activation movement"); ax.grid(alpha=.25)
    if len(records)<=10: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"layer_displacement.png",dpi=160); plt.close(fig)

    probes=load_probes(a.artifacts)
    if probes:
        fig,ax=plt.subplots(figsize=(10,6))
        for r in records:
            ls=[]; ss=[]
            for l,q in sorted(probes.items()):
                ls.append(l); ss.append(float(np.dot(r["activations"][l],q["weight"])+q["bias"]))
            ax.plot(ls,ss,marker="o",label=r["prompt"][:45])
        for l,q in sorted(probes.items()): ax.axhline(q["threshold"],linestyle="--",alpha=.4)
        ax.set(xlabel="Decoder layer",ylabel="Exp017 score",title="Exp017 unsafe-intent projection"); ax.grid(alpha=.25)
        if len(records)<=10: ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/"exp017_scores.png",dpi=160); plt.close(fig)

    print("\n[done]",out.resolve())
    print("Files: raw_activations.npz, trajectory_coordinates.npz, pca_projection.npz, prompts.json,")
    print("       trajectory_2d.png, trajectory_3d.png, activation_norm.png, layer_displacement.png")
    if probes: print("       exp017_scores.png")

if __name__=="__main__": main()
