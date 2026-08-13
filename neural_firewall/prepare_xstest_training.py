from pathlib import Path
import numpy as np

ROOT = Path("phase1_real/xstest")

for split in ["train", "calibration", "held_out_test"]:
    d = ROOT / split

    arrays = {
        str(layer): np.load(d / f"layer{layer}.npy")
        for layer in [19, 20, 21, 22]
    }

    np.savez_compressed(
        d / "activations.npz",
        **arrays,
    )

    print(
        split,
        "activations:",
        {k: v.shape for k, v in arrays.items()},
    )

print("Done.")