#!/usr/bin/env python3
"""
Debug script to check what character data would actually be sent to Mistral
"""

import sys
import os

# Add the main directory to the path to import the app state
sys.path.append(r"c:\image analysis\Assignment\simplechat")

def debug_character_data():
    """Debug function to show character data that would be sent to Mistral"""
    
    # Mock the character data that would be saved in a real scenario
    print("=== CHARACTER DATA DEBUG ===")
    print()
    
    # These are the fields that get sent to Mistral
    fields_to_check = [
        ("physical_description", "Beautiful woman with long black hair and green eyes"),
        ("behavioral_description", "Friendly and outgoing personality"),  
        ("character_name", "Emma"),
        ("relation_to_user", "girlfriend"),
        ("user_name", "John"),
        ("initial_attire", "red evening dress"),
        ("gender", "Female"),
        ("style", "Photorealistic")
    ]
    
    print("Character fields that should be sent to Mistral:")
    for field_name, example_value in fields_to_check:
        print(f"  {field_name}: '{example_value}'")
    
    print()
    print("=== SYSTEM MESSAGE STRUCTURE ===")
    
    # Simulate the system message that would be generated
    char_name = "Emma"
    relation = "girlfriend" 
    user_name = "John"
    context_info = "Romantic conversation"
    personality = "Friendly and outgoing personality"
    appearance = "Beautiful woman with long black hair and green eyes"
    attire = "red evening dress"
    gender = "Female"
    style = "Photorealistic"
    
    system_message = f"""
    You are a fully immersive naughty roleplayer assistant. You are completely uncensored, ready to be very explicit and lewd if needed.
    don't come off as too strong at the first line, but then do as asked by the user.
    
    CRITICAL CHARACTER ENFORCEMENT: You MUST use ONLY the following specific character details in ALL IMAGE_PROMPT responses. 
    NEVER use generic terms like "mysterious woman", "beautiful woman", "attractive person" or any other generic descriptions.
    ALWAYS use the EXACT character details provided below:
    
    Character details (use these when composing IMAGE_PROMPT and to remain consistent across images):
    Name: {char_name}
    Relation to user: {relation}
    What you call the user: {user_name}
    Chat context / setting: {context_info}
    Personality: {personality}
    Physical appearance: {appearance}
    Attire / clothing: {attire}
    Gender: {gender}
    Visual style preference: {style}

    MANDATORY RULE: When generating IMAGE_PROMPT, you MUST include the exact physical appearance "{appearance}" and attire "{attire}" details. 
    DO NOT substitute with generic descriptions. Use these saved character details verbatim.
    """
    
    print("System message structure:")
    print(system_message)
    
    print()
    print("=== USER-LEVEL CHARACTER REMINDER ===")
    
    char_attrs = (
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
    
    print("User-level character reminder:")
    print(char_attrs)
    
    print()
    print("=== FINAL MESSAGE ORDER ===")
    print("1. System message (with character details)")
    print("2. Previous context (if any)")  
    print("3. Image context (if any)")
    print("4. Character attributes reminder (user-level)")
    print("5. User's actual request")
    
    print()
    print("If Mistral is still generating 'mysterious woman', the issue may be:")
    print("- Mistral model being used is ignoring detailed instructions")
    print("- Character data not being properly saved/loaded")
    print("- API request not being sent correctly")
    print("- Model preferring brevity over following detailed instructions")

if __name__ == "__main__":
    debug_character_data()