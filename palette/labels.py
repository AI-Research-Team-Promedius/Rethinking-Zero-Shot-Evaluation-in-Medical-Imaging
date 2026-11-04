"""PALETTE label definitions and the zero-shot prompt template.

The label-exclusive split is derived from the CXR-LT 2024 label file (see make_split.py):

  * frontal views only (ViewPosition in {PA, AP});
  * EXCLUDED_CLASSES are dropped as label columns (their images are kept);
  * of the remaining 40 classes, the 12 with the fewest positive images are HELD_OUT_CLASSES;
    every image positive for any of them goes to the zero-shot evaluation set;
  * the other 28 classes are SEEN_CLASSES; the remaining images are split 8:2 by study_id.

The lists below are the result of that procedure on CXR-LT 2024. make_split.py recomputes
the held-out set from the data and checks it against HELD_OUT_CLASSES.
"""

EXCLUDED_CLASSES = ["Bulla", "Cardiomyopathy", "Hilum", "Osteopenia", "Scoliosis"]

# 12 held-out (zero-shot) classes, "cls12".
HELD_OUT_CLASSES = [
    "Lobar Atelectasis", "Round(ed) Atelectasis", "Clavicle Fracture", "Azygos Lobe",
    "Pleural Other", "Pneumoperitoneum", "Kyphosis", "Hydropneumothorax", "Infarction",
    "Pneumomediastinum", "Pulmonary Hypertension", "Fibrosis",
]

# 28 seen classes, "cls28".
SEEN_CLASSES = [
    "Adenopathy", "Atelectasis", "Calcification of the Aorta", "Cardiomegaly",
    "Consolidation", "Edema", "Emphysema", "Enlarged Cardiomediastinum",
    "Fissure", "Fracture", "Granuloma", "Hernia", "Infiltration", "Lung Lesion",
    "Lung Opacity", "Mass", "Nodule", "Normal", "Pleural Effusion", "Pleural Thickening",
    "Pneumonia", "Pneumothorax", "Pulmonary Embolism", "Rib Fracture",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta", "Tuberculosis",
]

# Positive / negative prompt pair. The score for a class is
# softmax over (sim(image, "<class>"), sim(image, "no <class>")), positive entry.
PROMPT_TEMPLATE = ("{}", "no {}")

# Non-label columns of the CXR-LT 2024 label file.
META_COLUMNS = ["dicom_id", "subject_id", "study_id", "ViewPosition",
                "ViewCodeSequence_CodeMeaning", "fpath", "split"]

TASKS = {"cls12": HELD_OUT_CLASSES, "cls28": SEEN_CLASSES}
