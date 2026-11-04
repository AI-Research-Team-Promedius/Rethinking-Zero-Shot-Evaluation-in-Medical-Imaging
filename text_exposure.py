"""How much training-report text mentions each held-out (cls12) class?

PALETTE removes held-out classes from the training *labels*; the training *reports* are not
filtered. This script measures how often the training impressions mention a held-out class
anyway, so the residual text exposure can be reported next to zero-shot results.

It reads only training-side files (labels_cls28_train.csv, impressions_cls28_train.csv from
make_split.py). Reports are paired with images as a CLIP-style loader would: one impression per
study, left-joined onto the image rows. Counts are per study and per image.

Two tiers per class:
  exact  the literal class name used in the zero-shot prompt
  broad  synonyms radiologists write (judgement calls; patterns below)

Each matching sentence is classed affirmative / negated / uncertain with NegEx-style cue rules
(a pre-negation cue counts only within the 10 words before the term and never across a clause
break). A study is affirmative if any sentence is affirmative, else uncertain, else negated.
Run tests/test_negation_rules.py after changing any rule.

  python text_exposure.py --labels data/palette/labels_cls28_train.csv \
      --impressions data/palette/impressions_cls28_train.csv --out_dir results/

--samples N writes N random matching sentences per class and status to the Markdown report for
spot-checking. These are report text: keep that file under your PhysioNet data use agreement and
do not publish it.
"""
import argparse
import json
import os
import random
import re

import pandas as pd

CLASSES = {
    # class: (exact prompt-string regex, broad synonym regex)
    "Lobar Atelectasis":      (r"lobar atelectasis",
                               r"lobar (?:atelectasis|collapse)|lobe (?:atelectasis|collapse)"),
    "Round(ed) Atelectasis":  (r"round\(ed\) atelectasis|rounded atelectasis|round atelectasis",
                               r"round(?:ed)? atelectasis|folded lung"),
    "Clavicle Fracture":      (r"clavicle fracture",
                               r"clavic\w* fracture|fracture[^.;]{0,30}clavic\w*"),
    "Azygos Lobe":            (r"azygos lobe",
                               r"azygos (?:lobe|fissure)"),
    "Pleural Other":          (r"pleural other",
                               r"pleural (?:plaque|calcification|scar\w*)"),   # approximate: no single text form
    "Pneumoperitoneum":       (r"pneumoperitoneum",
                               r"pneumoperitoneum|free (?:intraperitoneal |intra-abdominal |sub\w* )?air|subdiaphragmatic (?:free )?air"),
    "Kyphosis":               (r"kyphosis",
                               r"kyphos\w*|kyphotic"),          # not kypho\w*: that matched "kyphoplasty"
    "Hydropneumothorax":      (r"hydropneumothorax",
                               r"hydro-?pneumothora\w*"),
    "Infarction":             (r"infarction",
                               r"infarct\w*"),
    "Pneumomediastinum":      (r"pneumomediastinum",
                               r"pneumomediastinum|mediastinal (?:air|emphysema)"),
    "Pulmonary Hypertension": (r"pulmonary hypertension",
                               r"pulmonary (?:arterial |artery )?hypertension|\bpah\b|enlarged (?:central )?pulmonary arter\w*|pulmonary arter\w* (?:enlargement|dilat\w*)"),
    "Fibrosis":               (r"fibrosis",
                               r"fibro(?:sis|tic)|interstitial lung disease|\bild\b|honeycomb\w*|\buip\b"),
}

UNC_STRONG = re.compile(r"cannot be (?:excluded|ruled out)|can't be excluded|not (?:be )?excluded|"
                        r"\brule out\b|\br/o\b|evaluat\w* for|assess(?:ment)? for|question of")
# NegEx-style scope: a pre-negation cue only counts within the 10 words before the term, and
# never across a clause break. Whole-sentence scope lets an unrelated earlier "not" negate a
# later finding. Commas are NOT breaks: treating them as breaks un-negates the later items of a
# negated list ("no X, Y, or Z").
NEG_PRE = re.compile(r"\b(?:no|not|without|negative for|free of|absence of|resolved|resolution of|clear of)\b")
PSEUDO_NEG = re.compile(r"\b(?:no|without) (?:significant |substantial |interval |appreciable )?"
                        r"(?:change|increase|decrease|worsening|improvement)\b|\bnot necessarily\b")
CLAUSE = re.compile(r"\b(?:but|however|though|although|while|which|with|as well as|except)\b|:")
NEG_POST = re.compile(r"^[^.;]{0,40}\b(?:has resolved|have resolved|resolved|cleared|not seen|not identified|"
                      r"not present|no longer|is absent|ruled out)\b")
HEDGE = re.compile(r"\b(?:possibl\w*|may|might|could|suspici\w*|concern\w* for|equivocal|versus|vs|"
                   r"differential|likely|probabl\w*|suggest\w*|consider\w*)\b")
SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+|\s(?=\d+\.\s)")


def classify(sentence, m):
    if UNC_STRONG.search(sentence):
        return "uncertain"
    pre = CLAUSE.split(PSEUDO_NEG.sub(" ", sentence[:m.start()]))[-1]
    pre = " ".join(pre.split()[-10:])
    if NEG_PRE.search(pre) or NEG_POST.search(sentence[m.end():]):
        return "negated"
    if HEDGE.search(sentence):
        return "uncertain"
    return "affirmative"


def study_status(text, rx):
    """-> ('affirmative'|'uncertain'|'negated'|None, [(status, sentence)])"""
    hits = []
    for s in SPLIT.split(text):
        m = rx.search(s)
        if m:
            hits.append((classify(s, m), s.strip()))
    if not hits:
        return None, hits
    st = {h[0] for h in hits}
    return ("affirmative" if "affirmative" in st else "uncertain" if "uncertain" in st else "negated"), hits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="labels_cls28_train.csv from make_split.py")
    p.add_argument("--impressions", required=True, help="impressions_cls28_train.csv from make_split.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--samples", type=int, default=0,
                   help="random sentences per class/status to include for spot-checking (report text)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    txt = pd.read_csv(args.impressions)
    txt["study_id"] = txt["filename"].str.replace(".txt", "", regex=False)
    txt = txt.drop_duplicates(subset=["study_id"])
    lab = pd.read_csv(args.labels)
    merged = lab[["study_id"]].merge(txt[["study_id", "impression"]], on="study_id", how="left")
    imgs_per_study = merged.groupby("study_id").size()
    studies = txt[txt.study_id.isin(lab.study_id)].set_index("study_id")["impression"].fillna("").str.lower()
    print(f"training images {len(merged)}, unique studies with text {len(studies)}, "
          f"images with no text {merged.impression.isna().sum()}")

    rng = random.Random(0)
    results, samples = {}, {}
    for cls, (exact, broad) in CLASSES.items():
        results[cls] = {}
        for tier, pat in (("exact", exact), ("broad", broad)):
            rx = re.compile(pat)
            status, sents = {}, {"affirmative": [], "uncertain": [], "negated": []}
            for sid, t in studies.items():
                st, hits = study_status(t, rx)
                if st:
                    status[sid] = st
                    for h in hits:
                        sents[h[0]].append(h[1])
            row = {}
            for cat in ("affirmative", "uncertain", "negated"):
                ids = [s for s, v in status.items() if v == cat]
                row[cat] = {"studies": len(ids), "images": int(imgs_per_study.reindex(ids).fillna(0).sum())}
            results[cls][tier] = row
            if tier == "broad" and args.samples:
                samples[cls] = {c: rng.sample(v, min(args.samples, len(v))) for c, v in sents.items()}

    with open(os.path.join(args.out_dir, "text_exposure.json"), "w") as fh:
        json.dump({"n_studies": len(studies), "n_images": len(merged), "results": results,
                   "patterns": CLASSES}, fh, indent=2)

    md = ["# Held-out class mentions in the training reports", "",
          f"{len(studies):,} unique training studies / {len(merged):,} training images.", "",
          "**exact** = the literal prompt string; **broad** = synonyms (patterns in `text_exposure.py`).",
          "Counts are **studies** (images in parentheses). A study counts as affirmative if any "
          "sentence mentions the class without a negation or hedge cue.", "",
          "| Class | exact: affirm | exact: uncertain | exact: negated | broad: affirm | broad: uncertain | broad: negated |",
          "|---|---|---|---|---|---|---|"]
    for cls, r in results.items():
        f = lambda t, c: f"{r[t][c]['studies']:,} ({r[t][c]['images']:,})"
        md.append(f"| {cls} | " + " | ".join(f(t, c) for t in ("exact", "broad")
                                              for c in ("affirmative", "uncertain", "negated")) + " |")
    if samples:
        md += ["", "## Spot-check sentences (random, seed 0, broad tier)", "",
               "Report text: do not publish this section.", ""]
        for cls, d in samples.items():
            md.append(f"### {cls}")
            for cat, ss in d.items():
                for s in ss:
                    md.append(f"- *{cat}*: {s[:200]}")
            md.append("")
    with open(os.path.join(args.out_dir, "TEXT_EXPOSURE.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md[:9 + len(results)]))


if __name__ == "__main__":
    main()
