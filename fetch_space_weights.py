"""Fetch the served artifacts from the Hub model repo into models/ at image build time.

Keeping strategy_catboost.cbm out of the Space repo means one published copy of the
weights, and the Space always serves exactly what the model repo advertises.
"""
import pathlib
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "Veeda241/tessellate-f1-strategy"
FILES = ("strategy_catboost.cbm", "model_metadata.json")

target = pathlib.Path(__file__).resolve().parent / "models"
target.mkdir(exist_ok=True)

for name in FILES:
    source = hf_hub_download(repo_id=REPO_ID, filename=name)
    shutil.copyfile(source, target / name)
    print(f"fetched {name} -> {(target / name).stat().st_size} bytes")
