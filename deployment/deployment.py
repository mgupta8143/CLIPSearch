from pathlib import Path
import numpy as np
import pandas as pd
import clip
import torch
#import boto3

#s3 = boto3.client('s3')

## AWS Lambda Handler
def get_images_from_query(event, context):
    search_query = event["search_query"]
    num_images = event["num_images"]

    bucket = "clip_images"
    features_path = "features/features.npy"

    features_path = Path("./features")
    photo_features = np.load(features_path / "features.npy")
    photo_ids = pd.read_csv(features_path / "photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    text_features = text_encoded.cpu().numpy()
    similarities = list((text_features @ photo_features.T).squeeze(0))
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

    res = []
    for i in range(num_images):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        res.append(photo_id)

    return res
    

