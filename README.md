# PALETTE

**P**rotocol for **A**ssessing **L**abel-**E**xclusive zero-sho**T** **T**esting in m**E**dical imaging

Code for the PALETTE evaluation protocol from *Rethinking Zero-Shot Evaluation in Medical
Imaging*.

Under PALETTE, a model is tested zero-shot on classes it has never seen as a training
label. The 12 rarest classes of CXR-LT 2024 are held out, and every image positive for any of them
is removed from training and used as the zero-shot test set.

This repository has everything needed to apply the protocol to your own model:

- build the split from PhysioNet data
- the class lists and prompts
- the evaluation (per-class AUROC)
- a check of how often held-out classes are still mentioned in the training reports

It contains **no** training code, model weights, images, reports or labels.

## Contents

- [Protocol](#protocol)
- [Setup](#setup)
- [Step by step](#step-by-step)
- [Checking your split against ours](#checking-your-split-against-ours)
- [Evaluating your model](#evaluating-your-model)
- [Repository layout](#repository-layout)
- [Citation](#citation) · [License](#license)

## Protocol

```
CXR-LT 2024 labels (MIMIC-CXR, 45 classes)
  │
  ├─ keep frontal views (PA, AP)
  ├─ drop 5 classes as label columns (images kept)
  ├─ 12 rarest of the remaining 40 = held out (cls12); other 28 = seen (cls28)
  │
  ├─ image positive for ANY held-out class ──────────► cls12_zero_shot   (test, zero-shot)
  │
  └─ all other images ── 8:2 by study_id, seed 42 ──┬─► cls28_train      (training)
                                                     └─► cls28_test       (checkpoint selection)
```

1. Keep frontal views only (`ViewPosition` PA or AP).
2. Drop Bulla, Cardiomyopathy, Hilum, Osteopenia and Scoliosis as label columns, to match the
   40-class setup of the CXR-CML baseline. These are the 5 zero-shot classes of CXR-LT 2024
   (Task 3). Their images stay in the pool.
3. Of the remaining 40 classes, the 12 with the fewest positive images are **held out**
   (`cls12`). The other 28 are **seen** (`cls28`). Both lists are in `palette/labels.py`.
4. Every image positive for any held-out class goes to the zero-shot test set.
5. The remaining images are split 8:2 by `study_id` (seed 42) into `cls28_train` and
   `cls28_test`.

**How to use the sets.** Train on `cls28_train`, using its images and report impressions;
no training label is a held-out class. Select checkpoints on `cls28_test`. Report zero-shot
results on `cls12_zero_shot`.

**Prompts.** For class *c*, an image's score is the softmax over its cosine similarity to the
text embeddings of `"c"` and `"no c"`, positive entry, as in CheXzero. Class names are used
exactly as written in `palette/labels.py`, e.g. `"Round(ed) Atelectasis"`.

**Metric.** AUROC per held-out class, and their unweighted mean.

### Split sizes

| Set | Images | Studies | Subjects |
|---|---:|---:|---:|
| `cls28_train` | 189,660 | 170,002 | 56,227 |
| `cls28_test` | 47,462 | 42,501 | 24,590 |
| `cls12_zero_shot` | 6,212 | 5,636 | 3,666 |

Positive images per held-out class in `cls12_zero_shot` (an image can be positive for more
than one class):

| Class | Positives | Class | Positives |
|---|---:|---|---:|
| Clavicle Fracture | 144 | Hydropneumothorax | 683 |
| Round(ed) Atelectasis | 151 | Infarction | 692 |
| Azygos Lobe | 152 | Pneumomediastinum | 841 |
| Lobar Atelectasis | 176 | Pulmonary Hypertension | 854 |
| Kyphosis | 470 | Fibrosis | 1,042 |
| Pleural Other | 505 | | |
| Pneumoperitoneum | 625 | | |

### What the split does not control

- **Patients.** PALETTE excludes labels, not patients. Images are assigned one by one, and the
  train/test split is by study. 2,943 of the 3,666 zero-shot subjects also have images in
  `cls28_train`.
- **Report text.** Training reports are not filtered, so they can still mention a held-out
  finding. `text_exposure.py` counts how often this happens.

## Setup

**Data.** You need your own credentialed PhysioNet access to:

| Dataset | Used for |
|---|---|
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) | images |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | radiology reports (impressions) |
| [CXR-LT 2024](https://physionet.org/content/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays/2.0.0/) | `labels.csv` (the 45 classes) |

**Environment.** The code was tested with Python 3.10 and the versions in `requirements.txt`. A GPU
is recommended for evaluation but not required.

```bash
pip install -r requirements.txt
```

**Disk.** `cxr.h5` (all MIMIC-CXR images at 320×320) is about 155 GB. The three split files
add about 100 GB (`cls28_train` 78 GB, `cls28_test` 19 GB, `cls12_zero_shot` 2.5 GB).

## Step by step

### 1. Preprocess images and reports

```bash
python preprocess_mimic.py \
    --images_dir  /path/to/mimic-cxr-jpg/files \
    --reports_dir /path/to/mimic-cxr-reports/files/ \
    --out_dir data/
```

Same preprocessing as CheXzero: each image is resized with its aspect ratio kept and zero-padded
to 320×320, and the IMPRESSION section of each report is extracted.
Writes `cxr_paths.csv`, `cxr.h5` and `mimic_impressions.csv`.

### 2. Build the split

```bash
python make_split.py \
    --labels      /path/to/cxr-lt-2024/labels.csv \
    --cxr_paths   data/cxr_paths.csv \
    --impressions data/mimic_impressions.csv \
    --images      data/cxr.h5 \
    --out_dir     data/palette/
```

For each set (`cls28_train`, `cls28_test`, `cls12_zero_shot`), this writes:

| File | Contents |
|---|---|
| `labels_<set>.csv` | label rows, one per image |
| `impressions_<set>.csv` | report impressions for the studies in the set |
| `images_<set>.h5` | images (dataset `cxr`), same row order as `labels_<set>.csv` |

It also writes `split_ids.csv`, with the image IDs and set of every image. Add `--skip_images` to write only the CSVs.
The script prints the held-out classes it derived, and warns if they differ from
`palette/labels.py`.

### 3. Evaluate

```bash
python evaluate.py --checkpoint model.pt --task cls12 \
    --images data/palette/images_cls12_zero_shot.h5 \
    --labels data/palette/labels_cls12_zero_shot.csv \
    --out_dir results/
```

Prints the AUROC per class and the mean. With `--out_dir`, it also writes `predictions.npy`
(images × classes) and `auroc.json`. See [Evaluating your model](#evaluating-your-model) for
the checkpoint format.

### 4. (Optional) Measure text exposure

```bash
python text_exposure.py \
    --labels      data/palette/labels_cls28_train.csv \
    --impressions data/palette/impressions_cls28_train.csv \
    --out_dir results/
python tests/test_negation_rules.py      # rule regression tests
```

For each held-out class, counts the training studies and images whose impressions mention the
class affirmatively, uncertainly or as a negation. Each class is counted twice: once by its
exact class name ("exact") and once with a wider set of synonyms ("broad"). The patterns are
listed in the script. `--samples N` adds example sentences for spot-checking. Those sentences
are report text, so keep that output under your data use agreement and don't publish it.

## Checking your split against ours

`split_ids/split_ids.csv.gz` lists `dicom_id`, `study_id`, `subject_id` and set for every image
of the split used in the paper. It contains identifiers only, no labels or text.

```python
import pandas as pd
ours = pd.read_csv("split_ids/split_ids.csv.gz")
mine = pd.read_csv("data/palette/split_ids.csv")
key = ["dicom_id", "set"]
print(ours.sort_values(key)[key].reset_index(drop=True)
          .equals(mine.sort_values(key)[key].reset_index(drop=True)))
```

The zero-shot set depends only on the labels. The train/test split also depends on the row
order of `labels.csv`, so treat `split_ids` as the reference if the two disagree.

## Evaluating your model

`evaluate.py` takes a PyTorch state dict for the OpenAI CLIP / CheXzero architecture:

- **Architecture:** ViT or ResNet image encoder. Image size, width and depth are read from the
  weights.
- **Format:** the state dict itself, or a dict with the state dict under `"model"`. A
  `module.` prefix (DataParallel) is stripped.
- **Extra keys** (for example, heads used only in training) are ignored and listed. A missing
  CLIP key is an error.
- **Ensembles:** pass several `--checkpoint` paths to average their predicted probabilities.
- **Seen classes:** `--task cls28` with the `cls28_test` files evaluates the 28 seen classes.

**Safe loading.** Checkpoints are loaded with `torch.load(weights_only=True)`, which reads
tensors only. A file that also pickles other objects (optimizer state, numpy scalars) is
refused, because unpickling an untrusted file can execute code. Either save just the state
dict:

```python
torch.save(model.state_dict(), "model.pt")
```

or pass `--trust_checkpoint` for a file you created yourself.

Images are normalised with the MIMIC-CXR pixel statistics used by CheXzero, then resized to the
model's input resolution.

## Repository layout

```
make_split.py            build the PALETTE split
evaluate.py              zero-shot evaluation
text_exposure.py         held-out class mentions in training reports
preprocess_mimic.py      MIMIC-CXR images/reports -> h5 / csv
palette/labels.py        class lists, excluded classes, prompt template
palette/model.py         CLIP model             ┐
palette/clip.py          CLIP tokenizer API     │ from CheXzero / OpenAI CLIP (MIT)
palette/simple_tokenizer.py, bpe_*.txt.gz       │
palette/data_process.py  image/report helpers   ┘
split_ids/               identifiers of the paper's split
tests/                   negation-rule tests
```

## Citation

To be added on publication.

## License

MIT, see `LICENSE`. This repository includes code from
[CheXzero](https://github.com/rajpurkarlab/CheXzero) and
[OpenAI CLIP](https://github.com/openai/CLIP), both MIT; see `NOTICE.md`.
