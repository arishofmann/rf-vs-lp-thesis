"""Split embeddings into train/val/test from metadata.parquet."""

import argparse
from pathlib import Path

import pandas as pd
import torch


METADATA_PATH = "./data/metadata.parquet"
BASE_EMBEDDING_DIR = "./embeddings"


def split_country(country, metadata_path, base_dir):
    country_dir = Path(base_dir) / country.lower()

    embeddings = torch.load(country_dir / "embeddings.pt", weights_only=True)
    labels = torch.load(country_dir / "labels.pt", weights_only=True)
    patch_ids = torch.load(country_dir / "patch_ids.pt", weights_only=False)

    print(f"{country}: {len(patch_ids)} patches, embeddings {embeddings.shape}, labels {labels.shape}")

    df = pd.read_parquet(metadata_path)
    df_country = df[df["country"] == country][["patch_id", "split"]]

    pid_to_split = dict(zip(df_country["patch_id"], df_country["split"]))

    split_indices = {"train": [], "validation": [], "test": []}
    missing = 0
    for i, pid in enumerate(patch_ids):
        s = pid_to_split.get(pid)
        if s in split_indices:
            split_indices[s].append(i)
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing} patch_ids not found in metadata")

    for split_name, indices in split_indices.items():
        idx = torch.tensor(indices, dtype=torch.long)
        out_dir = country_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(embeddings[idx], out_dir / "embeddings.pt")
        torch.save(labels[idx], out_dir / "labels.pt")
        torch.save([patch_ids[i] for i in indices], out_dir / "patch_ids.pt")

        print(f"  {split_name}: {len(indices)} patches -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split embeddings by train/val/test")
    parser.add_argument("--countries", nargs="+", default=["Finland", "Portugal"])
    parser.add_argument("--metadata", default=METADATA_PATH)
    parser.add_argument("--base-dir", default=BASE_EMBEDDING_DIR)
    args = parser.parse_args()

    for country in args.countries:
        split_country(country, args.metadata, args.base_dir)


if __name__ == "__main__":
    main()
