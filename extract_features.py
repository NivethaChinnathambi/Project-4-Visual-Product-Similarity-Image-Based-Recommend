"""
Feature Extraction Script - SIMPLE VERSION FOR FLAT DIRECTORY
Extracts 2048-dimensional embeddings from product images using ResNet50
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

METADATA_FILE = 'data/metadata.csv'
OUTPUT_EMBEDDINGS = 'data/embeddings.npy'
OUTPUT_IMAGE_IDS = 'data/image_ids.npy'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================

class FeatureExtractor:
    """Extracts features using ResNet50"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_model(self):
        """Load pre-trained ResNet50 model"""
        print("Loading pre-trained ResNet50 model...")
        model = models.resnet50(pretrained=True)
        # Remove the classification layer to get 2048-dim features
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
    
    def extract_features(self, image_batch):
        """Extract features from a batch of images"""
        with torch.no_grad():
            features = self.model(image_batch)
            # Flatten to 2048-dim vector
            features = features.view(features.size(0), -1)
        return features.cpu().numpy()

# ============================================================================
# MAIN EXTRACTION PROCESS
# ============================================================================

def extract_all_features():
    """Extract features from all product images"""
    
    print("=" * 70)
    print("FEATURE EXTRACTION - ResNet50")
    print("=" * 70)
    
    # Load metadata
    print(f"\nLoading metadata from {METADATA_FILE}...")
    df = pd.read_csv(METADATA_FILE)
    print(f"Total images: {len(df)}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(device=DEVICE)
    print(f"Using device: {DEVICE}")
    
    # Extract features
    all_embeddings = []
    all_product_ids = []
    success_count = 0
    error_count = 0
    
    print(f"\nExtracting features...")
    print("-" * 70)
    
    for idx, row in df.iterrows():
        # Use image_path directly from metadata
        image_path = row['image_path']
        product_id = row['product_id']
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = extractor.transform(image)
            image = image.unsqueeze(0).to(DEVICE)
            
            # Extract features
            features = extractor.extract_features(image)
            
            all_embeddings.append(features)
            all_product_ids.append(product_id)
            success_count += 1
            
            # Progress
            if (idx + 1) % 50 == 0:
                print(f"Processed: {idx + 1}/{len(df)} images")
                
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # Only print first 3 errors
                print(f"Error processing {image_path}: {e}")
            continue
    
    print("-" * 70)
    
    if len(all_embeddings) == 0:
        print("❌ ERROR: No images were successfully processed!")
        return None, None
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    print(f"\n✅ Extraction complete!")
    print(f"Successfully processed: {success_count}/{len(df)} images")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Save embeddings
    print(f"\nSaving embeddings to {OUTPUT_EMBEDDINGS}...")
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    
    print(f"Saving product IDs to {OUTPUT_IMAGE_IDS}...")
    np.save(OUTPUT_IMAGE_IDS, np.array(all_product_ids))
    
    print("\n" + "=" * 70)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_EMBEDDINGS} {embeddings.shape}")
    print(f"  - {OUTPUT_IMAGE_IDS} ({len(all_product_ids)} IDs)")
    
    return embeddings, all_product_ids

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    embeddings, product_ids = extract_all_features()
    
    if embeddings is not None:
        # Print sample
        print(f"\nSample embeddings (first 3 images):")
        print(f"  Shape: {embeddings[:3].shape}")
        print(f"  Product IDs: {product_ids[:3]}")
        print(f"  First embedding (first 10 values): {embeddings[0][:10]}")
