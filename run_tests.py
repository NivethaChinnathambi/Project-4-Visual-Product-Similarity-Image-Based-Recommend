"""
Comprehensive Test Suite
Tests all modules and functionality
"""

import os
import sys
import traceback
from pathlib import Path

# ============================================================================
# TEST SUITE
# ============================================================================

class TestSuite:
    """Run comprehensive tests"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test_data_files_exist(self):
        """Test 1: Check if all data files exist"""
        print("\n📋 TEST 1: Data Files Existence")
        print("-" * 70)
        
        required_files = [
            'data/metadata.csv',
            'data/embeddings.npy',
            'data/image_ids.npy',
            'index/faiss_index.bin'
        ]
        
        all_exist = True
        for filepath in required_files:
            if os.path.exists(filepath):
                print(f"✅ {filepath}")
                self.passed += 1
            else:
                print(f"❌ {filepath} - NOT FOUND")
                self.failed += 1
                all_exist = False
        
        return all_exist
    
    def test_python_modules_import(self):
        """Test 2: Check if all Python modules can be imported"""
        print("\n📋 TEST 2: Python Module Imports")
        print("-" * 70)
        
        modules = [
            ('extract_features', 'Feature extraction module'),
            ('build_index', 'Index building module'),
            ('search_engine', 'Search engine module'),
            ('app', 'Streamlit app'),
            ('evaluate', 'Evaluation module')
        ]
        
        all_imported = True
        for module_name, description in modules:
            try:
                __import__(module_name)
                print(f"✅ {module_name}: {description}")
                self.passed += 1
            except Exception as e:
                print(f"❌ {module_name}: {str(e)[:50]}")
                self.failed += 1
                all_imported = False
        
        return all_imported
    
    def test_search_engine_functionality(self):
        """Test 3: Test search engine functionality"""
        print("\n📋 TEST 3: Search Engine Functionality")
        print("-" * 70)
        
        try:
            from search_engine import SearchEngine
            
            print("Loading search engine...")
            engine = SearchEngine()
            print("✅ Search engine loaded successfully")
            self.passed += 1
            
            print("Checking metadata...")
            if hasattr(engine, 'metadata') and len(engine.metadata) > 0:
                print(f"✅ Metadata loaded: {len(engine.metadata)} products")
                self.passed += 1
            else:
                print("❌ Metadata not loaded")
                self.failed += 1
                return False
            
            print("Checking FAISS index...")
            if hasattr(engine, 'index') and engine.index is not None:
                print(f"✅ FAISS index loaded: {engine.index.ntotal} vectors")
                self.passed += 1
            else:
                print("❌ FAISS index not loaded")
                self.failed += 1
                return False
            
            print("Checking feature extractor...")
            if hasattr(engine, 'extractor') and engine.extractor is not None:
                print("✅ Feature extractor initialized")
                self.passed += 1
            else:
                print("❌ Feature extractor not initialized")
                self.failed += 1
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            traceback.print_exc()
            self.failed += 1
            return False
    
    def test_image_files_exist(self):
        """Test 4: Check if image files exist"""
        print("\n📋 TEST 4: Image Files")
        print("-" * 70)
        
        image_dir = 'data/images'
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"✅ Image directory exists")
            print(f"✅ Found {len(images)} images")
            self.passed += 2
            
            if len(images) == 500:
                print("✅ Correct number of images (500)")
                self.passed += 1
                return True
            else:
                print(f"⚠️  Expected 500 images, found {len(images)}")
                return False
        else:
            print(f"❌ Image directory not found: {image_dir}")
            self.failed += 1
            return False
    
    def test_results_files(self):
        """Test 5: Check if evaluation results exist"""
        print("\n📋 TEST 5: Evaluation Results")
        print("-" * 70)
        
        results_files = [
            'results/evaluation_metrics.csv',
            'results/evaluation_metrics.png'
        ]
        
        all_exist = True
        for filepath in results_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ {filepath} ({size} bytes)")
                self.passed += 1
            else:
                print(f"⚠️  {filepath} - Not found (optional)")
        
        return all_exist
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 70)
        print("COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        
        self.test_data_files_exist()
        self.test_python_modules_import()
        self.test_search_engine_functionality()
        self.test_image_files_exist()
        self.test_results_files()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Total: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed")
        
        print("=" * 70)

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    suite = TestSuite()
    suite.run_all_tests()
