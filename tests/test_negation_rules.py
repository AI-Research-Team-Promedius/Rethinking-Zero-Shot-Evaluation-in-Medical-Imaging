"""Regression cases for the negation / uncertainty rules in text_exposure.py.

The sentences are written for this test (not taken from any report). Each targets a failure
mode of a simpler rule: whole-sentence negation scope, commas treated as clause breaks,
pseudo-negations ("no change in ..."), post-negations ("... has resolved"), and hedges.

  python tests/test_negation_rules.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import text_exposure as t  # noqa: E402

CASES = [
    # (sentence, class, expected study status)
    # whole-sentence scope would negate these
    ("appearance is not typical of fluid overload, and there is a new right upper lobe collapse.", "Lobar Atelectasis", "affirmative"),
    ("no change in the old left clavicular fracture.", "Clavicle Fracture", "affirmative"),
    ("in the absence of earlier films: healed fracture of the right clavicle with callus.", "Clavicle Fracture", "affirmative"),
    ("the pneumomediastinum has nearly cleared; no pneumothorax.", "Pneumomediastinum", "negated"),
    ("limited evaluation for free air on this supine view.", "Pneumoperitoneum", "uncertain"),
    ("post kyphoplasty changes in the lower thoracic spine.", "Kyphosis", None),
    # negated lists: commas must not end the negation scope
    ("heart is enlarged but there is no edema, pneumothorax, pneumomediastinum or effusion.", "Pneumomediastinum", "negated"),
    ("no effusion or widening of the mediastinum or pneumomediastinum.", "Pneumomediastinum", "negated"),
    ("no free air under the diaphragm, pneumothorax, or pneumomediastinum.", "Pneumomediastinum", "negated"),
    ("no acute process or free air.", "Pneumoperitoneum", "negated"),
    ("no acute intrathoracic abnormality or evidence of free air below the diaphragm.", "Pneumoperitoneum", "negated"),
    # controls
    ("no acute abnormality, with a small amount of postoperative intraperitoneal free air.", "Pneumoperitoneum", "affirmative"),
    ("no pneumoperitoneum.", "Pneumoperitoneum", "negated"),
    ("resolution of right lower lobe atelectasis and effusion.", "Lobar Atelectasis", "negated"),
    ("right lower lobe collapse.", "Lobar Atelectasis", "affirmative"),
    ("possible pneumomediastinum.", "Pneumomediastinum", "uncertain"),
    ("a pulmonary infarct cannot be excluded.", "Infarction", "uncertain"),
    ("stable kyphotic deformity of the thoracic spine.", "Kyphosis", "affirmative"),
]


def main():
    fails = 0
    for s, c, want in CASES:
        got, _ = t.study_status(s, re.compile(t.CLASSES[c][1]))
        ok = got == want
        fails += not ok
        print(("OK  " if ok else "FAIL"), f"{c:22s} want={want!s:11s} got={got!s:11s} | {s[:70]}")
    print(f"{len(CASES) - fails}/{len(CASES)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
