import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

import terratorch
from terratorch.registry import BACKBONE_REGISTRY

METADATA_PATH = "./data/metadata.parquet"
S2_DIR = "./data/BigEarthNet-S2"
OUTPUT_DIR = "./embeddings/geobench2"

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

BATCH_SIZE = 64
NUM_WORKERS = 4
TARGET_SIZE = 224

# GEO-Bench-2 subsample sizes (from geobench_v2/generate_benchmark/benv2.py)
SUBSAMPLE = {"train": 20000, "validation": 4000, "test": 4000}
RANDOM_STATE = 24

CLASS_NAMES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

S2L2A_MEANS = [1390.458, 1503.317, 1718.197, 1853.910, 2199.100,
               2779.975, 2987.011, 3083.234, 3132.220, 3162.988,
               2424.884, 1857.648]
S2L2A_STDS  = [2106.761, 2141.107, 2038.973, 2134.138, 2085.321,
               1889.926, 1820.257, 1871.918, 1753.829, 1797.379,
               1434.261, 1334.311]


class ReBENDataset(Dataset):
    def __init__(self, data_dir, metadata_df, means, stds):
        self.data_dir = Path(data_dir)
        self.metadata = metadata_df.reset_index(drop=True)
        self.means = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        patch_id = row["patch_id"]
        # structure: S2_DIR / tile / patch_id / patch_id_B02.tif
        tile = "_".join(patch_id.split("_")[:-2])
        patch_dir = self.data_dir / tile / patch_id

        bands = []
        for band in BAND_NAMES:
            band_path = patch_dir / f"{patch_id}_{band}.tif"
            with rasterio.open(band_path) as src:
                band_data = src.read(1).astype(np.float32)
            bands.append(band_data)

        # resize all bands to same size (120x120, the 10m resolution)
        target_h, target_w = bands[1].shape  # B02 is 10m
        resized = []
        for b in bands:
            if b.shape != (target_h, target_w):
                t = torch.from_numpy(b).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
                resized.append(t.squeeze().numpy())
            else:
                resized.append(b)
        image = np.stack(resized, axis=0)

        image = (image - self.means) / self.stds

        image_tensor = torch.from_numpy(image)
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        labels = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for lbl in row["labels"]:
            if lbl in CLASS_NAMES:
                labels[CLASS_NAMES.index(lbl)] = 1.0

        return {
            "image": image_tensor,
            "label": torch.from_numpy(labels),
            "patch_id": patch_id,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "validation", "test"])
    parser.add_argument("--test-run", action="store_true",
                        help="Only process 100 patches")
    args = parser.parse_args()

    split = args.split

    df = pd.read_parquet(METADATA_PATH)
    df_split = df[df["split"] == split].copy()
    print(f"Full {split} split: {len(df_split)} patches")

    # subsample to match GEO-Bench-2
    n = SUBSAMPLE[split]
    df_split = df_split.sample(n=n, random_state=RANDOM_STATE)
    print(f"After subsampling: {len(df_split)} patches")

    if args.test_run:
        df_split = df_split.head(100)
        print(f"TEST RUN: {len(df_split)} patches")

    output_dir = Path(OUTPUT_DIR) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TerraMind backbone...")
    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        modalities=["S2L2A"],
        pretrained=True,
    )
    model = model.eval().cuda()
    print("Backbone loaded.")

    dataset = ReBENDataset(S2_DIR, df_split, S2L2A_MEANS, S2L2A_STDS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    all_embeddings = []
    all_labels = []
    all_patch_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {split}"):
            images = batch["image"].cuda()
            features = model(images)[-1]
            pooled = features.mean(dim=1)

            all_embeddings.append(pooled.cpu())
            all_labels.append(batch["label"])
            all_patch_ids.extend(batch["patch_id"])

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    torch.save(embeddings, output_dir / "embeddings.pt")
    torch.save(labels, output_dir / "labels.pt")
    torch.save(all_patch_ids, output_dir / "patch_ids.pt")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
