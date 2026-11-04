"""Build the PALETTE label-exclusive split from CXR-LT 2024 labels and MIMIC-CXR.

Inputs (all from your own PhysioNet download, see README):
  --labels        CXR-LT 2024 labels.csv
  --cxr_paths     cxr_paths.csv   (written by preprocess_mimic.py)
  --impressions   mimic_impressions.csv (written by preprocess_mimic.py)
  --images        cxr.h5          (written by preprocess_mimic.py)

Outputs in --out_dir, for each of cls28_train, cls28_test, cls12_zero_shot:
  labels_<set>.csv        label rows for the images in the set
  impressions_<set>.csv   report impressions for the studies in the set
  images_<set>.h5         images, dataset "cxr", same row order as labels_<set>.csv
and split_ids.csv (dicom_id, study_id, subject_id, set) for comparing against split_ids/.

Procedure:
  1. keep frontal views (PA, AP)
  2. drop the 5 EXCLUDED_CLASSES as label columns (images are kept)
  3. the 12 classes with the fewest positives are held out
  4. any image positive for a held-out class -> zero-shot set
  5. remaining images: drop those without an image in cxr.h5, then split 8:2 by study_id (seed 42)
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from palette.labels import EXCLUDED_CLASSES, HELD_OUT_CLASSES, META_COLUMNS, SEEN_CLASSES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="CXR-LT 2024 labels.csv")
    p.add_argument("--cxr_paths", required=True, help="cxr_paths.csv from preprocess_mimic.py")
    p.add_argument("--impressions", required=True, help="mimic_impressions.csv from preprocess_mimic.py")
    p.add_argument("--images", help="cxr.h5 from preprocess_mimic.py (not needed with --skip_images)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_held_out", type=int, default=12)
    p.add_argument("--skip_images", action="store_true", help="write CSVs only, no .h5 files")
    p.add_argument("--batch_size", type=int, default=2000)
    return p.parse_args()


def copy_h5_rows(src_path, dst_path, indices, batch_size=2000):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        s = src["cxr"]
        d = dst.create_dataset("cxr", shape=(len(indices),) + s.shape[1:], dtype=s.dtype)
        for start in tqdm(range(0, len(indices), batch_size), desc=Path(dst_path).name):
            for i, j in enumerate(indices[start:start + batch_size]):
                d[start + i] = s[j]


def main():
    args = parse_args()
    if not args.skip_images and not args.images:
        raise SystemExit("--images is required unless --skip_images is given")
    np.random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. frontal views only
    df = pd.read_csv(args.labels)
    df = df[df["ViewPosition"].isin(["PA", "AP"])].copy()
    print(f"frontal images: {len(df)}")

    # 2. drop excluded label columns (rows are kept)
    df = df.drop(columns=[c for c in EXCLUDED_CLASSES if c in df.columns])

    # 3. held-out classes = the n rarest remaining classes
    class_cols = [c for c in df.columns if c not in META_COLUMNS]
    counts = df[class_cols].sum().sort_values(kind="quicksort")
    held_out = counts.head(args.n_held_out).index.tolist()
    seen = [c for c in class_cols if c not in held_out]
    print("held-out classes (positives):")
    for c in held_out:
        print(f"  {c}: {int(counts[c])}")
    if set(held_out) != set(HELD_OUT_CLASSES) or set(seen) != set(SEEN_CLASSES):
        print("WARNING: the derived class lists differ from palette/labels.py; "
              "this split will not match the paper.")

    # 4. any held-out positive -> zero-shot set
    rare_mask = df[held_out].sum(axis=1) > 0
    df_zs, df_common = df[rare_mask].copy(), df[~rare_mask].copy()

    # map each image to its row in cxr.h5 and drop images that are not there
    paths = pd.read_csv(args.cxr_paths)["Path"]
    path_to_idx = {"files/" + p.split("files/")[1]: i for i, p in enumerate(paths) if "files/" in p}
    for d in (df_zs, df_common):
        d["h5_index"] = d["fpath"].map(path_to_idx)
    n_missing = int(df_zs["h5_index"].isna().sum() + df_common["h5_index"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} label rows have no image in cxr.h5 and are dropped")
    df_zs = df_zs.dropna(subset=["h5_index"]).astype({"h5_index": int})
    df_common = df_common.dropna(subset=["h5_index"]).astype({"h5_index": int})

    # 5. 8:2 split of the remaining images by study_id
    studies = df_common["study_id"].unique()
    np.random.shuffle(studies)
    n_train = int(len(studies) * args.train_ratio)
    train_studies = set(studies[:n_train])
    df_train = df_common[df_common["study_id"].isin(train_studies)].copy()
    df_test = df_common[~df_common["study_id"].isin(train_studies)].copy()

    imp = pd.read_csv(args.impressions)
    imp["study_id"] = imp["filename"].str.replace(".txt", "", regex=False)

    ids = []
    for name, d in (("cls28_train", df_train), ("cls28_test", df_test), ("cls12_zero_shot", df_zs)):
        print(f"{name}: {len(d)} images, {d['study_id'].nunique()} studies")
        d.drop(columns=["h5_index"]).to_csv(out / f"labels_{name}.csv", index=False)
        imp[imp["study_id"].isin(set(d["study_id"]))][["filename", "impression"]].to_csv(
            out / f"impressions_{name}.csv", index=False)
        if not args.skip_images:
            copy_h5_rows(args.images, out / f"images_{name}.h5", d["h5_index"].tolist(), args.batch_size)
        ids.append(d[["dicom_id", "study_id", "subject_id"]].assign(set=name))
    pd.concat(ids).to_csv(out / "split_ids.csv", index=False)

    # per-class positives in the zero-shot set (compare with the paper's Table 1 n)
    print("zero-shot set positives per held-out class:")
    for c in HELD_OUT_CLASSES:
        if c in df_zs.columns:
            print(f"  {c}: {int(df_zs[c].sum())}")


if __name__ == "__main__":
    main()
