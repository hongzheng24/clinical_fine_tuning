"""
Clinical note source -- Asclepius Synthetic Clinical Notes (Hugging Face).

A reputable, citable, purpose-built synthetic dataset:
    Kweon et al., "Publicly Shareable Clinical Large Language Model Built
    on Synthetic Clinical Notes" (2023), arXiv:2309.00237
    https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes

Notes were synthesized by GPT-3.5 from open-access PMC-Patients case
reports -- explicitly built to contain no real PHI, unlike MIMIC-IV, which
requires PhysioNet credentialing + a signed Data Use Agreement that can
take days to clear. This is the direct swap-in replacement for that: same
idea (real-feeling discharge-summary-style notes), zero access friction.

We cache a handful of notes to disk on first run (cached_notes.json) so
the pipeline doesn't depend on a live Hugging Face download every time you
demo it -- important if you're recording on wifi you don't trust. Delete
the cache file (or pass force_refresh=True) to pull a fresh sample.
"""

import json
import os
from datasets import load_dataset

DATA_PATH = 'starmpcc/Asclepius-Synthetic-Clinical-Notes'
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cached_notes.json")
N_NOTES = 3


def _download_notes(n: int) -> dict[str, str]:
    dataset = load_dataset(DATA_PATH, split="train")

    notes, seen_ids = {}, set()
    for row in dataset:
        patient_id = row["patient_id"]
        if patient_id in seen_ids:
            continue  # dataset has multiple QA rows per note; keep one copy of each
        seen_ids.add(patient_id)
        notes[f"asclepius_{patient_id}"] = row["patient"]
        if len(notes) >= n:
            break
    return notes


def load_notes(n: int = N_NOTES, force_refresh: bool = False) -> dict[str, str]:
    if not force_refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Downloading {n} notes from Hugging Face (starmpcc/Asclepius-Synthetic-Clinical-Notes)...")
    notes = _download_notes(n)
    with open(CACHE_PATH, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"Cached to {CACHE_PATH}")
    return notes


if __name__ == "__main__":
    # `python data/clinical_notes.py` -- populate/refresh the cache standalone
    notes = load_notes(force_refresh=True)
    for name, text in notes.items():
        print(f"\n=== {name} ({len(text)} chars) ===\n{text[:300]}...")
