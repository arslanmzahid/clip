"""
Robust CLIP Image Indexer - Memory Safe Version
This version includes proper error handling and memory cleanup to avoid segfaults.
"""

import pickle
import gc
import sys
import os
from typing import List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Force CPU usage to avoid GPU memory issues
DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def load_model():
    """Load CLIP model and processor safely"""
    print("🔄 Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("✅ Model loaded successfully")
    return model, processor

def process_single_image(image_path: str, model, processor):
    """Process a single image safely with error handling"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            embedding = model.get_image_features(**inputs).cpu()
        
        # Clean up immediately
        del inputs
        gc.collect()
        
        return embedding
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def embed_images(image_dir: str) -> Tuple[np.ndarray, List[str]]:
    """Embed all images in directory with proper memory management"""
    model, processor = load_model()
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(SUPPORTED_FORMATS)]
    
    if not image_files:
        raise SystemExit(f"❌ No images found in {image_dir}")
    
    print(f"📁 Found {len(image_files)} images")
    
    paths, embeddings = [], []
    
    for i, fname in enumerate(image_files, 1):
        path = os.path.join(image_dir, fname)
        print(f"🔄 Processing {i}/{len(image_files)}: {fname}")
        
        embedding = process_single_image(path, model, processor)
        if embedding is not None:
            paths.append(path)
            embeddings.append(embedding.numpy())
        
        # Memory cleanup every 10 images
        if i % 10 == 0:
            gc.collect()
    
    if not embeddings:
        raise SystemExit("❌ No images could be processed")
    
    # Convert to numpy array
    embeddings_matrix = np.vstack(embeddings).astype('float32')
    
    # Clean up model
    del model, processor
    gc.collect()
    
    print(f"✅ Successfully processed {len(embeddings)} images")
    return embeddings_matrix, paths

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index with cosine similarity using manual normalization"""
    print("🔄 Creating FAISS index...")
    
    # Ensure data is the right type and contiguous
    embeddings = embeddings.astype(np.float32)
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)
    
    print(f"📊 Embedding shape: {embeddings.shape}")
    
    # Manual L2 normalization (safer than faiss.normalize_L2)
    print("🔄 Normalizing embeddings manually for cosine similarity...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms
    print("✅ Manual normalization completed")
    
    dimension = embeddings_normalized.shape[1]
    print(f"📐 Creating cosine similarity index with dimension: {dimension}")
    
    try:
        # Use Inner Product with normalized vectors = cosine similarity
        # Higher scores = better matches
        index = faiss.IndexFlatIP(dimension)
        print("✅ Index created successfully (cosine similarity)")
        
        # Add normalized embeddings
        print("🔄 Adding normalized embeddings to index...")
        index.add(embeddings_normalized)
        print(f"✅ Added {index.ntotal} vectors to index")
        
    except Exception as e:
        print(f"❌ FAISS error: {e}")
        # Fallback: save embeddings as numpy array instead
        print("🔄 Falling back to numpy save...")
        np.save("embeddings.npy", embeddings_normalized)
        print("✅ Saved embeddings as embeddings.npy")
        return None
    
    return index

def main():
    """Main function with error handling"""
    try:
        img_dir = sys.argv[1] if len(sys.argv) > 1 else "examples"
        
        if not os.path.exists(img_dir):
            raise SystemExit(f"❌ Directory {img_dir} does not exist")
        
        print(f"🚀 Starting indexing for directory: {img_dir}")
        
        # Process images
        embeddings, paths = embed_images(img_dir)
        
        # Create index
        index = create_faiss_index(embeddings)
        
        # Save files
        print("💾 Saving index and paths...")
        
        if index is not None:
            faiss.write_index(index, "clip.index")
            print("✅ FAISS index saved as clip.index")
        
        with open("paths.pkl", "wb") as f:
            pickle.dump(paths, f)
        print("✅ Paths saved as paths.pkl")
        
        print(f"🎉 Success! Indexed {len(paths)} images")
        if index is not None:
            print("📄 Files created: clip.index, paths.pkl")
        else:
            print("📄 Files created: embeddings.npy, paths.pkl")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()