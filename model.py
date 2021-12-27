import torch 
import clip
import math 
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

photos_path = Path("./images/")
photo_files = list(photos_path.glob("*.jpg"))
print("Photos found: " + str(len(photo_files)))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_features(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    return photos_features.cpu().numpy()

batch_size = 16
features_path = Path("./features/")
batches = math.ceil(len(photo_files) / batch_size)

for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")
    batch_ids_path = Path(features_path / f"{i:010d}.csv")
    batch_features_path = Path(features_path /  f"{i:010d}.npy")
    if not batch_features_path.exists():
        try:
            batch_files = photo_files[i*batch_size : (i+1)*batch_size]
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)
            photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
        except:
            print(f'Problem with batch {i}')


features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]
features = np.concatenate(features_list)
np.save(features_path / "features.npy", features)
photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
photo_ids.to_csv(features_path / "photo_ids.csv", index=False)