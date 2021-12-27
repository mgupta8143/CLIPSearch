from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import clip
import torch
import requests 

features_path = Path("./features")
dataset_path = Path("./unsplash-lite")

photos = pd.read_csv(dataset_path / "photos.tsv000", sep='\t', header=0)

photo_features = np.load(features_path / "features.npy")
photo_ids = pd.read_csv(features_path / "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_images_from_query(search_query, num_images):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    #could be improved by using a heap instead for larger sets
    text_features = text_encoded.cpu().numpy()
    similarities = list((text_features @ photo_features.T).squeeze(0))
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

    res = []
    for i in range(num_images):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        im = Image.open("./images/" + photo_id + ".jpg")
        res.append(im)

    return res

def display_images(image_list):
    for image in image_list:
        image.show()

images = get_images_from_query("", 3)
display_images(images)



