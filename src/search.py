"""
Robust CLIP Image Search - Memory Safe Version
This version includes proper error handling and memory cleanup.
"""

import sys
import pickle
import os
import gc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

# Force CPU usage to avoid GPU memory issues
DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

def load_model():
    """Load CLIP model and processor safely"""
    print("🔄 Loading CLIP model for text search...")
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("✅ Model loaded successfully")
    return model, processor

def text_embed(query: str, model, processor):
    """Generate text embedding safely"""
    try:
        inputs = processor(text=[query], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize for similarity search
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Clean up
        del inputs
        gc.collect()
        
        return text_features.cpu().numpy().astype("float32")
        
    except Exception as e:
        print(f"❌ Error generating text embedding: {e}")
        return None

def load_index_and_paths():
    """Load FAISS index and image paths safely"""
    try:
        if not os.path.exists("clip.index"):
            raise FileNotFoundError("❌ clip.index not found. Run build_index.py first!")
        
        if not os.path.exists("paths.pkl"):
            raise FileNotFoundError("❌ paths.pkl not found. Run build_index.py first!")
        
        print("📂 Loading FAISS index...")
        index = faiss.read_index("clip.index")
        
        print("📂 Loading image paths...")
        with open("paths.pkl", "rb") as f:
            paths = pickle.load(f)
        
        print(f"✅ Loaded index with {index.ntotal} images")
        return index, paths
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None

def search_images(query: str, k: int = 5):
    """Search for similar images"""
    try:
        # Load model
        model, processor = load_model()
        
        # Load index and paths
        index, paths = load_index_and_paths()
        if index is None or paths is None:
            return
        
        # Generate query embedding
        print(f"🔍 Searching for: '{query}'")
        query_embedding = text_embed(query, model, processor)
        if query_embedding is None:
            return
        
        # Perform search
        k = min(k, len(paths))  # Don't search for more results than we have
        distances, indices = index.search(query_embedding, k)
        
        # Clean up model
        del model, processor
        gc.collect()
        
        # Display results
        print(f"\n🎯 Top {k} matches for: '{query}'\n")
        print(f"{'Similarity':<10} {'Image Path'}")
        print("-" * 50)
        
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                # Convert cosine similarity to percentage (higher = better match)
                similarity_percent = score * 100
                print(f"{similarity_percent:.1f}%     {paths[idx]}")
        
        return distances[0], [paths[idx] for idx in indices[0] if idx != -1]
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None, None

def main():
    """Main function with error handling"""
    try:
        # Get query from command line arguments
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = "a red car"
            print(f"💡 No query provided, using default: '{query}'")
        
        # Perform search
        results = search_images(query)
        
        if results[0] is not None:
            print(f"\n✅ Search completed successfully!")
        else:
            print(f"\n❌ Search failed!")
            
    except KeyboardInterrupt:
        print("\n❌ Search interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()