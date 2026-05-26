"""
Build FAISS Index
Creates a FAISS index from embeddings for fast similarity search
"""

import numpy as np
import faiss
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDINGS_FILE = 'data/embeddings.npy'
IMAGE_IDS_FILE = 'data/image_ids.npy'
INDEX_DIR = 'index'
INDEX_FILE = os.path.join(INDEX_DIR, 'faiss_index.bin')

# ============================================================================
# BUILD INDEX
# ============================================================================

def build_faiss_index():
    """Build FAISS index from embeddings"""
    
    print("=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)
    
    # Create index directory if it doesn't exist
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Load embeddings
    print(f"\nLoading embeddings from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Load image IDs
    print(f"Loading image IDs from {IMAGE_IDS_FILE}...")
    image_ids = np.load(IMAGE_IDS_FILE)
    print(f"Image IDs shape: {image_ids.shape}")
    
    # Ensure embeddings are float32 (required by FAISS)
    embeddings = embeddings.astype(np.float32)
    
    # Create FAISS index
    print(f"\nCreating FAISS index...")
    
    # Use L2 (Euclidean) distance index
    # For cosine similarity, we could normalize embeddings and use L2
    # But L2 on normalized vectors = cosine distance
    
    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    print(f"Index dimension: {dimension}")
    print(f"Adding {len(embeddings)} vectors to index...")
    index.add(embeddings)
    
    print(f"Index is trained: {index.is_trained}")
    print(f"Index ntotal: {index.ntotal}")
    
    # Save index
    print(f"\nSaving index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    print("\n" + "=" * 70)
    print("✅ FAISS INDEX CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nIndex file: {INDEX_FILE}")
    print(f"File size: {os.path.getsize(INDEX_FILE)} bytes")
    
    return index, embeddings, image_ids

# ============================================================================
# TEST INDEX
# ============================================================================

def test_index(index, embeddings, image_ids):
    """Test the index with a sample query"""
    
    print("\n" + "=" * 70)
    print("TESTING INDEX")
    print("=" * 70)
    
    # Use first embedding as query
    query_embedding = embeddings[0:1]
    
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    print(f"Query product ID: {image_ids[0]}")
    
    # Search for top 5 similar images
    k = 5
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nTop {k} similar images:")
    print("-" * 70)
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"{i+1}. Product ID: {image_ids[idx]}, Distance: {dist:.4f}")
    
    print("-" * 70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    index, embeddings, image_ids = build_faiss_index()
    test_index(index, embeddings, image_ids)
    
    print("\n✅ All done! Index is ready for similarity search.")

