"""Convert MIMIC-CXR-JPG images and MIMIC-CXR reports to the files make_split.py reads.

Same preprocessing as CheXzero (run_preprocess.py, --dataset_type mimic): every image is
resized with its aspect ratio kept, zero-padded to 320x320 and stored in one HDF5 file; the
IMPRESSION section of each report is extracted.

Outputs:
  cxr_paths.csv          path of every image (row i = row i of cxr.h5)
  cxr.h5                 dataset "cxr", shape (N, 320, 320)
  mimic_impressions.csv  filename (s<study_id>.txt), impression
"""
import argparse
from pathlib import Path

from palette.data_process import get_cxr_path_csv, get_cxr_paths_list, img_to_hdf5, write_report_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True,
                   help="the files/ folder of MIMIC-CXR-JPG 2.x")
    p.add_argument("--reports_dir", required=True,
                   help="the files/ folder of the MIMIC-CXR reports (ends with a slash)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports_dir = args.reports_dir if args.reports_dir.endswith("/") else args.reports_dir + "/"

    get_cxr_path_csv(str(out / "cxr_paths.csv"), args.images_dir)
    cxr_paths = get_cxr_paths_list(str(out / "cxr_paths.csv"))
    img_to_hdf5(cxr_paths, str(out / "cxr.h5"))
    write_report_csv(cxr_paths, reports_dir, str(out / "mimic_impressions.csv"))


if __name__ == "__main__":
    main()
