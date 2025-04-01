"""
Script to check for Hugging Face tokens in settings.json file.
This script is used as a pre-commit hook to prevent committing sensitive tokens.
"""

import json
import sys
import os

def checkHfToken(filepath: str) -> int:
    """
    Check if settings.json contains a Hugging Face token.
    
    Args:
        file_path: Path to the settings.json file
        
    Returns:
        int: 1 if token is found, 0 if no token is found
    """
    try:
        with open(filepath, 'r') as file:
            settings: dict[str, str] = json.load(file)
            
        PossibleHfToken: str = settings.get('huggingfaceToken', '')
        
        if PossibleHfToken.startswith("hf_"):
            print(f"Error: Found Hugging Face token in {filepath}")
            print("Please remove or replace the token before committing.")
            return 1
            
        return 0
        
    except Exception as error:
        print(f"Error reading {filepath}: {error}")
        return 1


def main() -> int:
    """
    Main entry point for the script.
    
    Returns:
        int: Exit code (1 for failure, 0 for success)
    """
    settingsPath: str = "ytscript/settings.json"
    
    if not os.path.exists(settingsPath):
        print(f"Error: {settingsPath} not found")
        return 1
        
    return checkHfToken(settingsPath)


if __name__ == "__main__":
    sys.exit(main())