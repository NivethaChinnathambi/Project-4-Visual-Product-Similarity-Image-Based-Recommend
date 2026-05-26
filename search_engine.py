"""
Search Engine Module
Performs similarity search on FAISS index to find similar products
"""

import numpy as np
import faiss
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_FILE = 'index/faiss_index.bin'
EMBEDDINGS_FILE = 'data/embeddings.npy'
IMAGE_IDS_FILE = 'data/image_ids.npy'
METADATA_FILE = 'data/metadata.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================

class FeatureExtractor:
    """Extracts features from images using ResNet50"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_model(self):
        """Load pre-trained ResNet50 model"""
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
            features = features.view(features.size(0), -1)
        
        return features.cpu().numpy()

# ============================================================================
# SEARCH ENGINE CLASS
# ============================================================================

class SearchEngine:
    """Search engine for finding similar products"""
    
    def __init__(self):
        """Initialize search engine"""
        print("Initializing Search Engine...")
        
        # Load FAISS index
        self.index = faiss.read_index(INDEX_FILE)
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load image IDs
        self.image_ids = np.load(IMAGE_IDS_FILE)
        print(f"✅ Loaded {len(self.image_ids)} image IDs")
        
        # Load metadata
        self.metadata = pd.read_csv(METADATA_FILE)
        print(f"✅ Loaded metadata for {len(self.metadata)} images")
        
        # Initialize feature extractor
        self.extractor = FeatureExtractor(device=DEVICE)
        print(f"✅ Initialized feature extractor (device: {DEVICE})")
    
    def search_by_image(self, image_path, k=10):
        """
        Search for similar images given a query image path
        
        Args:
            image_path (str): Path to query image
            k (int): Number of results to return
            
        Returns:
            list: List of dicts with product info and distances
        """
        
        # Extract features from query image
        query_features = self.extractor.extract_features(image_path)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_features)
        
        # Search index
        distances, indices = self.index.search(query_features, k)
        
        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            product_id = self.image_ids[idx]
            metadata_row = self.metadata[self.metadata['product_id'] == product_id]
            
            if len(metadata_row) > 0:
                row = metadata_row.iloc[0]
                results.append({
                    'rank': i + 1,
                    'product_id': product_id,
                    'image_path': row['image_path'],
                    'category': row['category'],
                    'distance': float(dist),
                    'similarity_score': 1 - float(dist)  # Convert distance to similarity
                })
        
        return results
    
    def search_by_embedding(self, embedding, k=10):
        """
        Search for similar images given an embedding vector
        
        Args:
            embedding (np.array): Query embedding (1 x 2048)
            k (int): Number of results to return
            
        Returns:
            list: List of dicts with product info and distances
        """
        
        # Ensure embedding is float32
        embedding = embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)
        
        # Search index
        distances, indices = self.index.search(embedding, k)
        
        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            product_id = self.image_ids[idx]
            metadata_row = self.metadata[self.metadata['product_id'] == product_id]
            
            if len(metadata_row) > 0:
                row = metadata_row.iloc[0]
                results.append({
                    'rank': i + 1,
                    'product_id': product_id,
                    'image_path': row['image_path'],
                    'category': row['category'],
                    'distance': float(dist),
                    'similarity_score': 1 - float(dist)
                })
        
        return results

# ============================================================================
# TEST SEARCH ENGINE
# ============================================================================

def test_search_engine():
    """Test the search engine"""
    
    print("\n" + "=" * 70)
    print("TESTING SEARCH ENGINE")
    print("=" * 70)
    
    # Initialize search engine
    engine = SearchEngine()
    
    # Get first image from metadata
    first_image_path = 'data/images/backpack_0_1771054472729.jpg'
    
    if os.path.exists(first_image_path):
        print(f"\nSearching for similar images to: {first_image_path}")
        print("-" * 70)
        
        results = engine.search_by_image(first_image_path, k=10)
        
        print(f"\nTop 10 similar products:")
        print("-" * 70)
        for result in results:
            print(f"Rank {result['rank']}: {result['product_id']}")
            print(f"  Category: {result['category']}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
            print()
    else:
        print(f"Image not found: {first_image_path}")
    
    print("=" * 70)
    print("✅ Search engine test complete!")
    print("=" * 70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    test_search_engine()
