import pickle
from typing import List, Tuple
import sys
import os 

import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"


def embed_images(image_dir: str) -> Tuple[torch.Tensor, List[str]]:
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)  # loading the model 
    processor = CLIPProcessor.from_pretrained(MODEL_ID)  # loading the processor
    paths, embs = [], []
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(image_dir, fname)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu()
        paths.append(path)
        embs.append(emb)

    if not embs:  # incase embeddings are empty
        raise SystemExit(f"No images found in {image_dir}. Add some JPG/PNG files.")

    mat = torch.stack(embs).float()  # [N, D]
    return mat, paths


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "examples"
    mat, paths = embed_images(img_dir)

    index = faiss.IndexFlatIP(mat.shape[1])  # cosine if vectors are L2-normalized
    index.add(mat.numpy())
    faiss.write_index(index, "clip.index")
    with open("paths.pkl", "wb") as f:
        pickle.dump(paths, f)

    print(f"Indexed {len(paths)} images from {img_dir}")
    print("Wrote: clip.index, paths.pkl")


if __name__ == "__main__":
    main()
