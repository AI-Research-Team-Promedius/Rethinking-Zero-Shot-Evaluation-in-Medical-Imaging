# PALETTE: label-exclusive zero-shot evaluation for chest X-ray models

Code for the PALETTE evaluation protocol from *Rethinking Zero-Shot Evaluation in Medical
Imaging*. Every image carrying a held-out class is removed from training, so the
held-out classes are never seen as labels during training. Models are then tested zero-shot on
those classes.

This repository covers the protocol only: building the split, the prompts, the evaluation, and
a check of how often held-out classes appear in the training reports. It does not include
training code, model weights, or any data.

## The protocol

Source labels: CXR-LT 2024 (MIMIC-CXR images, 45 classes).

1. Keep frontal views only (`ViewPosition` PA or AP).
2. Drop 5 classes as label columns: Bulla, Cardiomyopathy, Hilum, Osteopenia, Scoliosis.
   Their images stay in the pool. <!-- TODO: one-line reason for the exclusion -->
3. Of the remaining 40 classes, the 12 with the fewest positive images are **held out**
   (`cls12`). The other 28 are **seen** (`cls28`).
4. Every image positive for any held-out class goes to the zero-shot test set.
5. The remaining images are split 8:2 by `study_id` (seed 42) into `cls28_train` and
   `cls28_test`.

Train on `cls28_train` (images and report impressions; the held-out classes appear in no
training label). Select checkpoints on `cls28_test`. Report zero-shot AUROC on `cls12_zero_shot`.

**Prompts.** For class *c*, an image's score is the softmax over its cosine similarity to
the text embeddings of `"c"` and `"no c"`, taking the positive entry (as in CheXzero). The class
names are used as written in `palette/labels.py`, e.g. `"Round(ed) Atelectasis"`.

**Metric.** Per-class AUROC on the zero-shot set, and their unweighted mean.

| Set | Images | Studies | Subjects |
|---|---|---|---|
| cls28_train | 189,660 | 170,002 | 56,227 |
| cls28_test | 47,462 | 42,501 | 24,590 |
| cls12_zero_shot | 6,212 | 5,636 | 3,666 |

Positives per held-out class in the zero-shot set:

| Class | n | Class | n |
|---|---|---|---|
| Round(ed) Atelectasis | 151 | Hydropneumothorax | 683 |
| Clavicle Fracture | 144 | Infarction | 692 |
| Azygos Lobe | 152 | Pneumomediastinum | 841 |
| Lobar Atelectasis | 176 | Pulmonary Hypertension | 854 |
| Pleural Other | 505 | Fibrosis | 1,042 |
| Pneumoperitoneum | 625 | Kyphosis | 470 |

**What the split does not control.**
- *Patients.* The protocol excludes labels, not patients. Selection is per image, and the
  train/test split is per study. 2,943 of the 3,666 zero-shot subjects also have images in
  `cls28_train`.
- *Report text.* Training reports are not filtered, so they can still mention a held-out
  finding. `text_exposure.py` measures how often this happens.

## Data access

MIMIC-CXR-JPG, MIMIC-CXR (reports) and CXR-LT 2024 are credentialed-access datasets on
PhysioNet. You need your own access. This repository contains no images, reports or labels.
`split_ids/split_ids.csv.gz` lists only identifiers (`dicom_id`, `study_id`, `subject_id`,
set), so you can check that your rebuilt split matches ours.

- MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/
- MIMIC-CXR (reports): https://physionet.org/content/mimic-cxr/
- CXR-LT 2024: https://physionet.org/content/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays/2.0.0/

## Usage

```bash
pip install -r requirements.txt

# 1. images -> cxr.h5 (320x320), report impressions -> mimic_impressions.csv  (as CheXzero)
python preprocess_mimic.py --images_dir /path/to/mimic-cxr-jpg/files \
    --reports_dir /path/to/mimic-cxr-reports/files/ --out_dir data/

# 2. build the split
python make_split.py --labels /path/to/cxr-lt-2024/labels.csv \
    --cxr_paths data/cxr_paths.csv --impressions data/mimic_impressions.csv \
    --images data/cxr.h5 --out_dir data/palette/

# 3. zero-shot evaluation of a CLIP-style checkpoint
python evaluate.py --checkpoint model.pt --task cls12 \
    --images data/palette/images_cls12_zero_shot.h5 \
    --labels data/palette/labels_cls12_zero_shot.csv --out_dir results/

# 4. (optional) held-out class mentions in the training reports
python text_exposure.py --labels data/palette/labels_cls28_train.csv \
    --impressions data/palette/impressions_cls28_train.csv --out_dir results/
python tests/test_negation_rules.py
```

`make_split.py --skip_images` writes the CSVs without copying images. To compare your split
with ours:

```python
import pandas as pd
ours = pd.read_csv("split_ids/split_ids.csv.gz")
mine = pd.read_csv("data/palette/split_ids.csv")
print(ours.sort_values("dicom_id").reset_index(drop=True)
          .equals(mine.sort_values("dicom_id").reset_index(drop=True)))
```

`evaluate.py` accepts a state dict of the OpenAI CLIP / CheXzero architecture (ViT or ResNet,
optionally under a `"model"` key) and reads the architecture from the weights. Keys that are not
part of CLIP, such as training-only heads, are ignored and listed. Pass several `--checkpoint`
paths to average their predictions. Use `--task cls28` with the `cls28_test` files to evaluate
the seen classes.

## Citation

To be added on publication.

## License

MIT (see `LICENSE`). Includes code from CheXzero and OpenAI CLIP, both MIT. See `NOTICE.md`.
