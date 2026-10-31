"""Fetches the three project repositories at their pinned commits into vendor/ (code, trained models, result files).
Nothing of them is copied into this repository: the analyzer always runs the exact published versions.
Run once after cloning:  python scripts/get_models.py"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holter.vendor import REPOS, VENDOR, url  # noqa: E402

NEEDED = {"beats": ["models/beatnet.pt", "results/end_to_end_ds2.csv"],
          "rhythm": ["models/afnet_ensemble.pt", "app/examples/examples.npz", "results/external_windows.csv"],
          "morphology": ["models/cnn_leadII_noise.pt", "models/cnn_leadII_sub23.pt", "models/quality_gate.joblib",
                         "models/thresholds.json", "results/sub23_leadII_auc.csv"]}


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(role):
    repo, sha, _ = REPOS[role]
    dst = os.path.join(VENDOR, repo)
    if os.path.isdir(os.path.join(dst, ".git")):
        try:
            if git("rev-parse", "HEAD", cwd=dst) == sha:
                return dst, "already there"
        except subprocess.CalledProcessError:
            pass
        git("fetch", "-q", "--depth", "1", "origin", sha, cwd=dst)
        git("checkout", "-q", sha, cwd=dst)
        return dst, "updated"
    os.makedirs(dst, exist_ok=True)
    git("init", "-q", cwd=dst)
    git("remote", "add", "origin", url(role) + ".git", cwd=dst)
    try:
        git("fetch", "-q", "--depth", "1", "origin", sha, cwd=dst)
        git("checkout", "-q", "FETCH_HEAD", cwd=dst)
    except subprocess.CalledProcessError:             # servers that refuse fetching by commit: full history
        git("fetch", "-q", "origin", cwd=dst)
        git("checkout", "-q", sha, cwd=dst)
    return dst, "downloaded"


def main():
    for role, (repo, sha, _) in REPOS.items():
        dst, how = fetch(role)
        missing = [f for f in NEEDED[role] if not os.path.exists(os.path.join(dst, f))]
        if missing:
            raise SystemExit(f"{repo} @ {sha[:7]} is missing {missing}")
        print(f"{repo:26s} @ {sha[:7]}  {how}")
    print(f"All three projects are in {VENDOR}")


if __name__ == "__main__":
    main()
