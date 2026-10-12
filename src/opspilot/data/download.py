"""Download the Olist CSVs from Kaggle into data/raw/ and print their checksums.

Run with `make data` or `python -m opspilot.data.download`.

The checksums are compared with data/raw.sha256 (committed), so everyone who
rebuilds the project starts from byte-identical files. Pass --update-checksums
to rewrite that file after a deliberate dataset version bump.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import kagglehub

from opspilot.config import KAGGLE_DATASET, RAW_CHECKSUMS, RAW_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> list[Path]:
    """Fetch the dataset into kagglehub's cache and copy the CSVs into data/raw/."""
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=force))
    sources = sorted(cache_dir.glob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(src, RAW_DIR / src.name)) for src in sources]


def read_checksums(path: Path) -> dict[str, str]:
    """Parse `sha256sum` format: one '<hash>  <filename>' per line."""
    checksums = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name] = digest
    return checksums


def write_checksums(path: Path, checksums: dict[str, str]) -> None:
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument(
        "--update-checksums",
        action="store_true",
        help=f"rewrite {RAW_CHECKSUMS.name} from the downloaded files",
    )
    args = parser.parse_args()

    files = download(force=args.force)
    actual = {f.name: sha256(f) for f in files}

    print(f"{len(files)} files in {RAW_DIR}")
    for f in files:
        print(f"{actual[f.name]}  {f.stat().st_size:>11,}  {f.name}")

    if args.update_checksums or not RAW_CHECKSUMS.exists():
        write_checksums(RAW_CHECKSUMS, actual)
        print(f"Wrote {RAW_CHECKSUMS}")
        return 0

    expected = read_checksums(RAW_CHECKSUMS)
    problems = [
        f"{name}: {'missing' if name not in actual else 'checksum mismatch'}"
        for name in expected
        if actual.get(name) != expected[name]
    ]
    problems += [f"{name}: not listed in {RAW_CHECKSUMS.name}" for name in actual.keys() - expected]
    if problems:
        print("Checksum verification FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"All checksums match {RAW_CHECKSUMS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
