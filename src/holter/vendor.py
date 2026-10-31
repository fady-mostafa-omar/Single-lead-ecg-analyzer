"""The three project repositories this analyzer builds on. scripts/get_models.py fetches them at the pinned commits
into vendor/; nothing of their code or models is copied into this repository."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENDOR = os.environ.get("HOLTER_VENDOR", os.path.join(ROOT, "vendor"))
USER = "fady-mostafa-omar"
REPOS = {   # role: (repository, pinned commit, package)
    "beats": ("ecg-arrhythmia-detector", "8909557eae9308712fd50456721ac8c9782b47a7", "ecgsuite"),
    "rhythm": ("ecg-af-detector", "4554887d24f69d546be17153bbf0321d70edf4cc", "afsuite"),
    "morphology": ("ecg-lead2-diagnosis", "246bed3a80ed3271b72020d0529647e968e6583d", "leadsuite"),
}


def path(role, *parts):
    return os.path.join(VENDOR, REPOS[role][0], *parts)


def url(role):
    return f"https://github.com/{USER}/{REPOS[role][0]}"


def activate():
    for role in REPOS:
        src = path(role, "src")
        if not os.path.isdir(src):
            raise FileNotFoundError(f"{src} is missing: run  python scripts/get_models.py")
        if src not in sys.path:
            sys.path.insert(0, src)
