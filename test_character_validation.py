#!/usr/bin/env python3
"""
Character Testing Tool - Run this to verify character attribute injection
"""

def test_character_injection_flow():
    """Test what happens when we have vs don't have character data"""
    
    print("=== CHARACTER INJECTION TEST ===\n")
    
    # Test Case 1: User has saved specific character
    print("TEST CASE 1: User has saved a specific character")
    print("-" * 50)
    
    # Sample saved character data
    saved_character = {
        'physical_description': 'Tall woman with curly red hair, freckles, and bright blue eyes',
        'behavioral_description': 'Adventurous and witty, loves exploring and making jokes',
        'character_name': 'Sarah',
        'relation_to_user': 'best friend',
        'user_name': 'Alex',
        'initial_attire': 'hiking boots, cargo pants, and a green jacket',
        'gender': 'Female',
        'style': 'Photorealistic'
    }
    
    def create_character_message(char_data):
        char_name = char_data.get('character_name', 'Fantasy Character')
        appearance = char_data.get('physical_description', 'A unique and mysterious figure')
        attire = char_data.get('initial_attire', 'appropriate clothing')
        personality = char_data.get('behavioral_description', 'A friendly and helpful character')
        gender = char_data.get('gender', 'unspecified')
        style = char_data.get('style', 'Photorealistic')
        
        return (
            f"CRITICAL REMINDER - This character has been saved with specific details. You MUST use these EXACT details in IMAGE_PROMPT:\n"
            f"Name: {char_name}\n"
            f"Physical appearance: {appearance}\n"
            f"Default attire: {attire}\n"
            f"Personality: {personality}\n"
            f"Gender: {gender}\n"
            f"Visual style: {style}\n\n"
            f"FORBIDDEN: Do NOT use generic terms like 'mysterious woman', 'beautiful woman', 'attractive person' or any vague descriptions. "
            f"Use the EXACT appearance details: '{appearance}' and attire: '{attire}' in your IMAGE_PROMPT."
        )
    
    char_message = create_character_message(saved_character)
    print("Character message sent to Mistral:")
    print(char_message)
    print("\nExpected IMAGE_PROMPT should include:")
    print("- 'Tall woman with curly red hair, freckles, and bright blue eyes'")
    print("- 'hiking boots, cargo pants, and a green jacket'")
    print("- NOT 'mysterious woman' or other generic terms")
    
    print("\n" + "="*70 + "\n")
    
    # Test Case 2: User has NOT saved character (defaults)
    print("TEST CASE 2: User has NOT saved character (using defaults)")
    print("-" * 50)
    
    default_character = {
        'physical_description': 'A unique and mysterious figure',
        'behavioral_description': 'A friendly and helpful character',
        'character_name': 'Fantasy Character',
        'relation_to_user': 'companion',
        'user_name': 'user',
        'initial_attire': 'appropriate clothing',
        'gender': 'unspecified',
        'style': 'Photorealistic'
    }
    
    default_message = create_character_message(default_character)
    print("Default character message sent to Mistral:")
    print(default_message)
    print("\nWith defaults, Mistral might generate generic descriptions")
    print("This is expected behavior when no specific character is saved")
    
    print("\n" + "="*70 + "\n")
    
    # Test Case 3: Check validation logic
    print("TEST CASE 3: Character validation logic")
    print("-" * 50)
    
    def validate_character_data(char_data):
        appearance = char_data.get('physical_description', '')
        has_specific_character = bool(
            appearance and 
            appearance != "A unique and mysterious figure" and
            len(appearance.strip()) > 10
        )
        return has_specific_character, appearance
    
    # Test with saved character
    is_specific, appearance = validate_character_data(saved_character)
    print(f"Saved character validation:")
    print(f"  Appearance: '{appearance}'")
    print(f"  Has specific data: {is_specific}")
    
    # Test with default character  
    is_specific_default, appearance_default = validate_character_data(default_character)
    print(f"\nDefault character validation:")
    print(f"  Appearance: '{appearance_default}'")
    print(f"  Has specific data: {is_specific_default}")
    
    print("\n" + "="*70 + "\n")
    print("DEBUGGING CHECKLIST:")
    print("1. Check console logs when user sends a message - does it show specific character data?")
    print("2. Look for 'Has specific character data: True/False' in debug output")
    print("3. Check if Mistral's raw response includes the specific appearance details")
    print("4. Verify that the character was actually saved (not just filled in the form)")

if __name__ == "__main__":
    test_character_injection_flow()