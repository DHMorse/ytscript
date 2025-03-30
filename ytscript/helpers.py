import os
import json
import warnings
import argparse

from constants import SETTINGS_JSON_FILEPATH, DEFAULT_SETTINGS

def defineArguments() -> argparse.Namespace:
    # Create argument parser
    parser = argparse.ArgumentParser(description="Demo for keyword arguments in CLI")

    # Define keyword arguments with -- prefix
    parser.add_argument("videoUrl", type=str, help="The URL of the video you want to transcribe")
    parser.add_argument("--filepath", type=str, required=False, help="The output file path of the mp3 and txt files")
    parser.add_argument("--keepMp3", type=str, required=False, help="Whether to keep the mp3 file after transcribing")
    parser.add_argument("--model", type=str, required=False, help="The model size you want to use, either 'tiny, 'base', 'small', 'medium', or 'large'")
    parser.add_argument("--summerizeationModelType", type=str, required=False, help="The model type you want to use for summerization, either 'local' or 'huggingface'")
    parser.add_argument("--huggingfaceToken", type=str, required=False, help="The token for the huggingface model")
    parser.add_argument("--summerize", type=str, required=False, help="Whether to open the OpenAI chat to summerize the text")
    parser.add_argument("--summerizeationModel", type=str, required=False, help="The model to use for summerization")
    parser.add_argument("--summerizeationPrompt", type=str, required=False, help="The prompt to use for summerization")

    # Parse arguments
    return parser.parse_args()

def checkForTrueOrFalse(value: str) -> bool | None:
    if value.lower() in ['true', 't', 'yes', 'y']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n']:
        return False
    else:
        return None

def settingsJsonToDict() -> dict[str, str]:
    settingsJsonDict: dict[str, str] = DEFAULT_SETTINGS
    settingsJson: dict[str, str]

    # Check if settings.json exists
    if not os.path.exists(SETTINGS_JSON_FILEPATH):
        warnings.warn("settings.json not found. Using default settings.")
        return settingsJsonDict

    # Check proper json format
    try:
        with open(SETTINGS_JSON_FILEPATH, 'r') as file:
            settingsJson = json.load(file)
    except json.JSONDecodeError:
        warnings.warn("settings.json is not a valid json file. Using default settings.")
        return settingsJsonDict

    # Check outputFilepath
    try:
        outputFilepath: str = settingsJson['outputFilepath']

        if not outputFilepath:
            warnings.warn("outputFilepath is not set. Using default setting.")
        elif not os.path.exists(outputFilepath):
            warnings.warn("outputFilepath does not exist. Using default setting.")
        else:
            settingsJsonDict['outputFilepath'] = outputFilepath
    except KeyError:
        warnings.warn("KeyError: outputFilepath is not set properly. Using default setting.")

    # Check Model
    try:
        model: str = settingsJson['model']
        
        if not model:
            warnings.warn("model is not set. Using default setting.")
        elif model not in ['tiny', 'base', 'small', 'medium', 'large']:
            warnings.warn("model is not set properly. Using default setting.")
        else:
            settingsJsonDict['model'] = model
    except KeyError:
        warnings.warn("KeyError: model is not set properly. Using default setting.")

    # Check Keep Mp3
    try:
        keepMp3: str = settingsJson['keepMp3']
        
        if not keepMp3:
            warnings.warn("keepMp3 is not set. Using default setting.")
        elif checkForTrueOrFalse(keepMp3) is None:
            warnings.warn("keepMp3 is not set properly. Using default setting.")
        else:
            settingsJsonDict['keepMp3'] = keepMp3
    except KeyError:
        warnings.warn("KeyError: keepMp3 is not set properly. Using default setting.")
    
    # Check summerize
    try:
        summerize: str = settingsJson['summerize']
        
        if not summerize:
            warnings.warn("summerize is not set. Using default setting.")
        elif checkForTrueOrFalse(summerize) is None:
            warnings.warn("summerize is not set properly. Using default setting.")
        else:
            settingsJsonDict['summerize'] = summerize
    except KeyError:
        warnings.warn("KeyError: summerize is not set properly. Using default setting.")

    # Summerizeation Model Type 
    try:
        summerizeationModelType: str = settingsJson['summerizeationModelType']

        if not summerizeationModelType:
            warnings.warn("summerizeationModelType is not set. Using default setting.")
        elif summerizeationModelType not in ['local', 'huggingface']:
            warnings.warn("summerizeationModelType is not set properly. Using default setting.")
        else:
            settingsJsonDict['summerizeationModelType'] = summerizeationModelType
    except KeyError:
        warnings.warn("KeyError: summerizeationModelType is not set properly. Using default setting.")

    # Check Huggingface Token
    try:
        huggingfaceToken: str = settingsJson['huggingfaceToken']
        
        if not huggingfaceToken:
            raise ValueError("huggingfaceToken is not set.")
        else:
            settingsJsonDict['huggingfaceToken'] = huggingfaceToken
    except KeyError:
        raise ValueError("KeyError: huggingfaceToken is not set properly.")

    # Check Summerizeation Model
    try:
        summerizeationModel: str = settingsJson['summerizeationModel']
        
        if not summerizeationModel:
            raise ValueError("summerizeationModel is not set.")
        else:
            settingsJsonDict['summerizeationModel'] = summerizeationModel
    except KeyError:
        raise ValueError("KeyError: summerizeationModel is not set properly.")

    # Check Summerizeation Prompt
    try:
        summerizeationPrompt: str = settingsJson['summerizeationPrompt']
        
        if not summerizeationPrompt:
            warnings.warn("summerizeationPrompt is not set. Using default setting.")
        else:
            settingsJsonDict['summerizeationPrompt'] = summerizeationPrompt
    except KeyError:
        warnings.warn("KeyError: summerizeationPrompt is not set properly. Using default setting.")

    return settingsJsonDict