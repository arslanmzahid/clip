"""
Robust CLIP Visual Search Streamlit UI
Memory-safe version with proper error handling
"""

import streamlit as st
import faiss
import pickle
import torch
import os
import gc
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Force CPU usage to avoid GPU memory issues
DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

@st.cache_resource
def load_all():
    """Load model, processor, index, and paths with error handling"""
    try:
        st.write("🔄 Loading CLIP model...")
        model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        st.write("✅ Model loaded successfully")
        
        if not os.path.exists("clip.index"):
            st.error("❌ clip.index not found! Please run: python src/build_index.py examples/")
            st.stop()
            
        if not os.path.exists("paths.pkl"):
            st.error("❌ paths.pkl not found! Please run: python src/build_index.py examples/")
            st.stop()
        
        st.write("📂 Loading search index...")
        index = faiss.read_index("clip.index")
        
        with open("paths.pkl", "rb") as f:
            paths = pickle.load(f)
        
        st.write(f"✅ Loaded {len(paths)} images")
        return model, processor, index, paths
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        st.error("Make sure you've run: python src/build_index.py examples/")
        st.stop()

def text_embed(query: str, model, processor):
    """Generate text embedding with error handling"""
    try:
        inputs = processor(text=[query], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize for cosine similarity
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Clean up
        del inputs
        gc.collect()
        
        return text_features.cpu().numpy().astype("float32")
        
    except Exception as e:
        st.error(f"❌ Error generating embedding: {e}")
        return None

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="CLIP Visual Search",
        page_icon="🔎",
        layout="wide"
    )
    
    st.title("🔎 CLIP Visual Search")
    st.write("Type a text query to find similar images using AI-powered semantic search!")
    
    # Load resources
    with st.spinner("Loading AI model and search index..."):
        model, processor, index, paths = load_all()
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Search Query", 
            value="a red sports car",
            placeholder="e.g., 'a cat', 'sunset over mountains', 'person with glasses'"
        )
    
    with col2:
        k = st.slider(
            "📊 Results to show", 
            min_value=1, 
            max_value=min(12, len(paths)), 
            value=min(5, len(paths))
        )
    
    # Search results
    if query and query.strip():
        with st.spinner(f"Searching for: '{query}'..."):
            # Generate query embedding
            query_embedding = text_embed(query.strip(), model, processor)
            
            if query_embedding is not None:
                # Perform search
                similarities, indices = index.search(query_embedding, k=k)
                
                # Display results
                st.subheader(f"🎯 Top {k} matches for: '{query}'")
                
                # Create columns for image grid
                cols_per_row = 3
                for i in range(0, k, cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j in range(cols_per_row):
                        idx_pos = i + j
                        if idx_pos < len(indices[0]):
                            img_idx = indices[0][idx_pos]
                            similarity = similarities[0][idx_pos]
                            img_path = paths[img_idx]
                            
                            with cols[j]:
                                try:
                                    # Load and display image
                                    image = Image.open(img_path)
                                    st.image(
                                        image, 
                                        caption=f"Similarity: {similarity*100:.1f}%\n{Path(img_path).name}",
                                        use_column_width=True
                                    )
                                    
                                    # Show file path in expander
                                    with st.expander("📁 Full path"):
                                        st.code(img_path)
                                        
                                except Exception as e:
                                    st.error(f"❌ Error loading {img_path}: {e}")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses **CLIP** (Contrastive Language-Image Pre-training) 
        to understand both images and text in the same vector space.
        
        **How it works:**
        1. Images are converted to embeddings using CLIP
        2. Your text query is converted to the same embedding space  
        3. Cosine similarity finds the most similar images
        """)
        
        st.header("📊 Statistics")
        st.metric("Total Images", len(paths))
        st.metric("Search Index Size", f"{index.ntotal} vectors")
        st.metric("Embedding Dimension", 512)
        
        st.header("🚀 Quick Start")
        st.code("""
# 1. Add images to examples/
# 2. Build index
python src/build_index.py examples/

# 3. Run UI
streamlit run src/app.py
        """)

if __name__ == "__main__":
    main()