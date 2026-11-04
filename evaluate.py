"""PALETTE zero-shot evaluation of a CLIP-style chest X-ray model.

For each class c, the score of an image is the softmax over its similarity to the prompts
"c" and "no c" (palette.labels.PROMPT_TEMPLATE), positive entry. The metric is per-class
AUROC, and the summary is the macro mean over the classes of the task.

  python evaluate.py --checkpoint model.pt \
      --images data/palette/images_cls12_zero_shot.h5 \
      --labels data/palette/labels_cls12_zero_shot.csv --task cls12 --out_dir results/

The checkpoint is a state dict of the OpenAI CLIP / CheXzero architecture (optionally under a
"model" key). The architecture is read from the state dict. Extra keys (e.g. heads used only
in training) are ignored and listed; a missing CLIP key is an error.
Several --checkpoint values are averaged (ensemble of predicted probabilities).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from tqdm import tqdm

from palette import clip
from palette.labels import PROMPT_TEMPLATE, TASKS
from palette.model import CLIP

# Pixel statistics of the MIMIC-CXR h5 images, as in CheXzero.
PIXEL_MEAN, PIXEL_STD = 101.48761, 83.43944


def clip_params_from_state_dict(sd):
    """Architecture hyper-parameters, as in model.build_model."""
    if "visual.proj" in sd:
        vision_width = sd["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in sd if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = sd["visual.conv1.weight"].shape[-1]
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid
    else:
        vision_layers = tuple(len({k.split(".")[2] for k in sd if k.startswith(f"visual.layer{b}")})
                              for b in (1, 2, 3, 4))
        vision_width = sd["visual.layer1.0.conv1.weight"].shape[0]
        vision_patch_size = None
        out_w = round((sd["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = out_w * 32
    transformer_width = sd["ln_final.weight"].shape[0]
    return dict(
        embed_dim=sd["text_projection"].shape[1],
        image_resolution=image_resolution, vision_layers=vision_layers,
        vision_width=vision_width, vision_patch_size=vision_patch_size,
        context_length=sd["positional_embedding"].shape[0],
        vocab_size=sd["token_embedding.weight"].shape[0],
        transformer_width=transformer_width, transformer_heads=transformer_width // 64,
        transformer_layers=len({k.split(".")[2] for k in sd if k.startswith("transformer.resblocks")}),
    )


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model = CLIP(**clip_params_from_state_dict(sd))
    expected = set(model.state_dict())
    missing = sorted(expected - set(sd))
    if missing:
        raise SystemExit(f"{path}: checkpoint is missing CLIP weights, e.g. {missing[:5]}")
    extra = sorted(set(sd) - expected - {"input_resolution", "context_length", "vocab_size"})
    if extra:
        prefixes = sorted({k.split(".")[0] for k in extra})
        print(f"ignoring {len(extra)} non-CLIP keys (prefixes: {', '.join(prefixes)})")
    model.load_state_dict({k: v for k, v in sd.items() if k in expected})
    return model.float().to(device).eval()


class H5Images(torch.utils.data.Dataset):
    def __init__(self, path, resolution):
        self.path, self.dset = path, None
        with h5py.File(path, "r") as f:
            self.n = len(f["cxr"])
        self.transform = Compose([
            Normalize((PIXEL_MEAN,) * 3, (PIXEL_STD,) * 3),
            Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.dset is None:  # open per worker
            self.dset = h5py.File(self.path, "r")["cxr"]
        img = torch.from_numpy(np.repeat(self.dset[i][None], 3, axis=0))
        return self.transform(img)


@torch.no_grad()
def text_embeddings(model, classnames, template, device):
    tokens = clip.tokenize([template.format(c) for c in classnames],
                           context_length=model.context_length).to(device)
    emb = model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)  # (n_classes, d)


@torch.no_grad()
def predict(model, loader, classnames, device):
    """Probability of the positive prompt, shape (n_images, n_classes)."""
    pos = text_embeddings(model, classnames, PROMPT_TEMPLATE[0], device)
    neg = text_embeddings(model, classnames, PROMPT_TEMPLATE[1], device)
    out = []
    for images in tqdm(loader, desc="images"):
        f = model.encode_image(images.to(device))
        f = f / f.norm(dim=-1, keepdim=True)
        s_pos = torch.nan_to_num(f @ pos.T, nan=0.0, posinf=100.0, neginf=-100.0)
        s_neg = torch.nan_to_num(f @ neg.T, nan=0.0, posinf=100.0, neginf=-100.0)
        diff = (s_neg - s_pos).cpu().numpy().astype(np.float64)
        out.append(1.0 / (1.0 + np.exp(np.clip(diff, -100, 100))))  # softmax, positive entry
    return np.nan_to_num(np.concatenate(out), nan=0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--images", required=True, help="images_<set>.h5 from make_split.py")
    p.add_argument("--labels", required=True, help="labels_<set>.csv from make_split.py")
    p.add_argument("--task", choices=sorted(TASKS), default="cls12")
    p.add_argument("--out_dir", help="write predictions.npy and auroc.json here")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classnames = TASKS[args.task]
    y_true = pd.read_csv(args.labels)[classnames].to_numpy()

    preds = []
    for path in args.checkpoint:
        model = load_model(path, device)
        ds = H5Images(args.images, model.visual.input_resolution)
        if len(ds) != len(y_true):
            raise SystemExit(f"{args.images} has {len(ds)} images but {args.labels} has {len(y_true)} rows")
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        preds.append(predict(model, loader, classnames, device))
    y_pred = np.mean(preds, axis=0)

    auroc = {c: float(roc_auc_score(y_true[:, i], y_pred[:, i])) for i, c in enumerate(classnames)}
    mean = float(np.mean(list(auroc.values())))
    for c, v in auroc.items():
        print(f"{c:24s} {v:.4f}")
    print(f"{'mean AUROC':24s} {mean:.4f}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "predictions.npy", y_pred)
        with open(out / "auroc.json", "w") as fh:
            json.dump({"task": args.task, "checkpoints": args.checkpoint,
                       "auroc": auroc, "mean_auroc": mean}, fh, indent=2)


if __name__ == "__main__":
    main()
