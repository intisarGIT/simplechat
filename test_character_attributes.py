#!/usr/bin/env python3
"""
Test script to verify character attribute injection logic
"""

class MockAppState:
    def __init__(self):
        self.physical_description = "Beautiful woman with long black hair, green eyes, wearing a red dress"
        self.behavioral_description = "Friendly and outgoing, loves to laugh and make jokes"
        self.character_name = "Emma"
        self.relation_to_user = "girlfriend"
        self.user_name = "John"
        self.initial_attire = "red evening dress"
        self.gender = "Female"

def test_character_injection():
    """Test the character attribute injection logic"""
    app_state = MockAppState()
    
    # Simulate the character attributes injection
    character_attrs = []
    
    if app_state.physical_description:
        character_attrs.append(f"Physical appearance: {app_state.physical_description}")
    if app_state.behavioral_description:
        character_attrs.append(f"Personality: {app_state.behavioral_description}")
    if app_state.character_name:
        character_attrs.append(f"Name: {app_state.character_name}")
    if app_state.relation_to_user:
        character_attrs.append(f"Relationship to user: {app_state.relation_to_user}")
    if app_state.initial_attire:
        character_attrs.append(f"Current attire: {app_state.initial_attire}")
    if app_state.gender:
        character_attrs.append(f"Gender: {app_state.gender}")
    
    if character_attrs:
        character_info = "Character Setup: " + "; ".join(character_attrs)
        print("DEBUG - Character attributes being injected:")
        print(character_info)
        print()
        
        # Simulate the message structure that would be sent to Mistral
        user_message = "Generate an IMAGE_PROMPT for the character in a garden"
        
        print("Message structure sent to Mistral:")
        print(f"1. Character Info: {character_info}")
        print(f"2. User Request: {user_message}")
        print()
        
        # Show what the final message would look like
        final_message = f"{character_info}\n\nUser request: {user_message}"
        print("Final combined message:")
        print(final_message)
    else:
        print("No character attributes found!")

if __name__ == "__main__":
    test_character_injection()