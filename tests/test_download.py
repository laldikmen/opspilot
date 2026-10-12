"""Checksum helpers used by the raw-data download (no network access)."""

import hashlib

from opspilot import config
from opspilot.data.download import read_checksums, sha256, write_checksums


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"order_id,status\n1,delivered\n")
    assert sha256(f) == hashlib.sha256(b"order_id,status\n1,delivered\n").hexdigest()


def test_checksum_file_round_trips(tmp_path):
    manifest = tmp_path / "raw.sha256"
    checksums = {"b.csv": "0" * 64, "a.csv": "f" * 64}
    write_checksums(manifest, checksums)
    assert read_checksums(manifest) == checksums
    assert manifest.read_text().splitlines()[0].endswith("a.csv")  # sorted by filename


def test_data_paths_live_inside_the_repo():
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    assert config.RAW_DIR.is_relative_to(config.REPO_ROOT)
    assert config.SAMPLE_DIR.is_relative_to(config.REPO_ROOT)
