"""Download the MIT-BIH Arrhythmia Database from PhysioNet into data/mitdb/.

Run from the repository root:  python tools/download_mitdb.py

The data is NOT part of this repository and must not be committed (data/ is in
.gitignore). Each person downloads it from PhysioNet themselves, which keeps
this repository free of third-party data.

MIT-BIH Arrhythmia Database v1.0.0, George Moody and Roger Mark.
Open Access, Open Data Commons Attribution License v1.0.
https://physionet.org/content/mitdb/1.0.0/
See NOTICE for the full credit and the citations PhysioNet asks for.
"""

from __future__ import annotations

from pathlib import Path

import wfdb

TARGET = Path(__file__).resolve().parent.parent / "data" / "mitdb"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(TARGET))
    n = len(list(TARGET.glob("*.dat")))
    print(f"Downloaded {n} records to {TARGET}")
    print("Remember: if you show results from this data, keep the notice from NOTICE next to them.")


if __name__ == "__main__":
    main()
