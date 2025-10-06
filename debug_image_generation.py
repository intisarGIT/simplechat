#!/usr/bin/env python3

"""
Debug script to test image generation without the full app
"""

import os
import sys

# Add the current directory to the path so we can import from chat10fixed6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the needed functions
from chat10fixed6 import generate_image, generate_image_pollinations, app_state

def test_image_generation():
    """Test image generation functions"""
    
    print("=== TESTING IMAGE GENERATION DEBUG ===")
    
    # Test prompt
    test_prompt = "A beautiful woman with long black hair, wearing a red dress"
    
    print(f"Current ImageRouter API key status: {'SET' if app_state.imagerouter_api_key else 'NOT SET'}")
    print(f"Current Pollinations API key status: {'SET' if app_state.pollinations_api_key else 'NOT SET'}")
    
    print("\n1. Testing ImageRouter API...")
    try:
        result_filename, result_message = generate_image(test_prompt)
        print(f"ImageRouter Result: filename='{result_filename}', message='{result_message}'")
    except Exception as e:
        print(f"ImageRouter Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing Pollinations API directly...")
    try:
        result_filename, result_message = generate_image_pollinations(test_prompt)
        print(f"Pollinations Result: filename='{result_filename}', message='{result_message}'")
    except Exception as e:
        print(f"Pollinations Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== END DEBUG TEST ===")

if __name__ == "__main__":
    test_image_generation()