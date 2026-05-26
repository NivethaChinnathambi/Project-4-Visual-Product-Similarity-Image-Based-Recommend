#!/usr/bin/env python3
"""
Verification script to ensure all dependencies are installed correctly.
"""

import sys

def verify_packages():
    """Check if all required packages are installed."""
    
    packages = {
        'torch': 'PyTorch (Deep Learning)',
        'torchvision': 'TorchVision (Pre-trained Models)',
        'faiss': 'FAISS (Similarity Search)',
        'streamlit': 'Streamlit (Web Framework)',
        'PIL': 'Pillow (Image Processing)',
        'numpy': 'NumPy (Numerical Computing)',
        'pandas': 'Pandas (Data Manipulation)',
        'sklearn': 'Scikit-Learn (ML Utilities)',
        'matplotlib': 'Matplotlib (Visualization)',
        'cv2': 'OpenCV (Computer Vision)'
    }
    
    print("=" * 70)
    print("DEPENDENCY VERIFICATION REPORT")
    print("=" * 70)
    
    all_installed = True
    
    for package_name, description in packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'Unknown')
            status = "✅ INSTALLED"
            print(f"{status} | {package_name:20} | v{version:15} | {description}")
        except ImportError:
            status = "❌ MISSING"
            print(f"{status} | {package_name:20} | {'':15} | {description}")
            all_installed = False
    
    print("=" * 70)
    
    if all_installed:
        print("\n✅ SUCCESS! All dependencies are installed correctly.")
        print("You are ready to proceed to Step 2!\n")
        return True
    else:
        print("\n❌ ERROR! Some packages are missing.")
        print("Please run: pip install -r requirements.txt\n")
        return False

if __name__ == "__main__":
    success = verify_packages()
    sys.exit(0 if success else 1)
