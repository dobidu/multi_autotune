#!/usr/bin/env python3
"""
Test script to check if SoundTouch is being detected correctly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_autotune import LibraryDetector, MultiLibraryAutotuneProcessor

def test_soundtouch_detection():
    """Test SoundTouch detection."""
    print("🔍 Testing SoundTouch detection...")
    
    # Test direct detection
    soundtouch_info = LibraryDetector._detect_soundtouch()
    print(f"SoundTouch Info: {soundtouch_info}")
    print(f"Available: {soundtouch_info.available}")
    print(f"Library Required: {soundtouch_info.library_required}")
    print(f"Installation Notes: {soundtouch_info.installation_notes}")
    
    # Test through processor
    print("\n🔍 Testing through processor...")
    processor = MultiLibraryAutotuneProcessor()
    available_methods = processor.get_available_methods()
    print(f"Available methods: {available_methods}")
    
    if "soundtouch_shift" in available_methods:
        print("✅ SoundTouch is available in the processor!")
    else:
        print("❌ SoundTouch is NOT available in the processor")
    
    # Test method info
    try:
        method_info = processor.get_method_info("soundtouch_shift")
        print(f"Method info: {method_info}")
    except Exception as e:
        print(f"Error getting method info: {e}")

if __name__ == "__main__":
    test_soundtouch_detection() 