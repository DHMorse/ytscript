import os
import json
import argparse
import warnings
from yt_dlp import YoutubeDL # type: ignore
from huggingface_hub import HfApi
from rich.panel import Panel
from rich.console import Console

from ytscript.constants import SETTINGS_JSON_FILEPATH, DEFAULT_SETTINGS

console = Console()

def richWarning(message: str, category: str, filename: str, lineno: int, line: str | None = None) -> str:
    panel = Panel(f"{message} (File: {filename}, Line: {lineno})", title="[bold yellow]Warning[/]")
    console.print(panel)
    return f"{message} (File: {filename}, Line: {lineno})"

warnings.formatwarning = richWarning

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
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    # Parse arguments
    return parser.parse_args()

def checkForTrueOrFalse(value: str) -> bool | None:
    if value.lower() in ['true', 't', 'yes', 'y']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n']:
        return False
    else:
        return None

def getTimeString(time: int | float) -> str:
    """
    Converts a time in seconds to a string in the format "HH:MM:SS".
    
    Args:
        time (int | float): The time in seconds.

    Returns:
        str: The time in the format "HH:MM:SS".
    """
    if type(time) == float:
        time = int(time)

    if time > 3600:
        return f'{time//3600:02d}:{(time%3600)//60:02d}:{time%60:02d}'
    elif time > 60:
        return f'00:{time//60:02d}:{time%60:02d}'
    else:
        return f'00:00:{time:02d}'

def getVideoLength(url: str) -> int:
    """
    Gets the length of a YouTube video in seconds.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        int: The length of the video in seconds.
    """
    with YoutubeDL() as ytDL:
        infoDict: dict = ytDL.extract_info(url, download=False)
        return infoDict.get('duration', 0)

def getVideoFilename(url: str) -> str:
    """
    Extracts the video filename from a YouTube URL.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: The filename of the video.
    """
    with YoutubeDL() as ytDL:
        infoDict: dict = ytDL.extract_info(url, download=False)
        videoTitle: str = infoDict.get('title', '')
        return videoTitle

def validateHuggingfaceToken(token: str) -> bool:
    """
    Validates if a Huggingface token is valid by attempting to use the API.
    
    Args:
        token (str): The Huggingface API token to validate
        
    Returns:
        bool: True if the token is valid, False otherwise
    """
    try:
        # Create an API client with the token
        api = HfApi(token=token)
        # Attempt a simple API call that requires authentication
        api.whoami()
        return True
    except Exception:
        return False

def validateHuggingfaceModel(model: str, token: str) -> bool:
    """
    Validates if a Huggingface model exists and is accessible with the given token.
    
    Args:
        model (str): The Huggingface model ID to validate
        token (str): The Huggingface API token to use for validation
        
    Returns:
        bool: True if the model exists and is accessible, False otherwise
    """
    try:
        # Create an API client with the token
        api = HfApi(token=token)
        
        # Try to get model info - this will fail if the model doesn't exist
        # or if the user doesn't have access to it
        api.model_info(model)
        return True
    except Exception:
        return False

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
            warnings.warn("model is not set. Set it to one of the following: tiny, base, small, medium, large. Using default setting.")
        elif model not in ['tiny', 'base', 'small', 'medium', 'large']:
            warnings.warn("model is not set properly. Set it to one of the following: tiny, base, small, medium, large. Using default setting.")
        else:
            settingsJsonDict['model'] = model
    except KeyError:
        warnings.warn("KeyError: model is not set properly. Set it to one of the following: tiny, base, small, medium, large. Using default setting.")

    # Check Keep Mp3
    try:
        keepMp3: str = settingsJson['keepMp3']
        
        if not keepMp3:
            warnings.warn("keepMp3 is not set. Set it to true or false. Using default setting.")
        elif checkForTrueOrFalse(keepMp3) is None:
            warnings.warn("keepMp3 is not set properly. Set it to true or false. Using default setting.")
        else:
            settingsJsonDict['keepMp3'] = keepMp3
    except KeyError:
        warnings.warn("KeyError: keepMp3 is not set properly. Set it to true or false. Using default setting.")
    
    # Check summerize
    try:
        summerize: str = settingsJson['summerize']
        
        if not summerize:
            warnings.warn("summerize is not set. Set it to true or false. Using default setting.")
        elif checkForTrueOrFalse(summerize) is None:
            warnings.warn("summerize is not set properly. Set it to true or false. Using default setting.")
        else:
            settingsJsonDict['summerize'] = summerize
    except KeyError:
        warnings.warn("KeyError: summerize is not set properly. Set it to true or false. Using default setting.")

    # Summerizeation Model Type 
    try:
        summerizeationModelType: str = settingsJson['summerizeationModelType']

        if not summerizeationModelType:
            warnings.warn("summerizeationModelType is not set. Set it to one of the following: local, huggingface. Using default setting.")
        elif summerizeationModelType not in ['local', 'huggingface']:
            warnings.warn("summerizeationModelType is not set properly. Set it to one of the following: local, huggingface. Using default setting.")
        else:
            settingsJsonDict['summerizeationModelType'] = summerizeationModelType
    except KeyError:
        warnings.warn("KeyError: summerizeationModelType is not set properly. Set it to one of the following: local, huggingface. Using default setting.")

    # Check Huggingface Token
    try:
        huggingfaceToken: str = settingsJson['huggingfaceToken']
        
        if not huggingfaceToken:
            warnings.warn("huggingfaceToken is not set. Set it to your Huggingface API token. Using default setting.")
        else:
            settingsJsonDict['huggingfaceToken'] = huggingfaceToken
    except KeyError:
        warnings.warn("KeyError: huggingfaceToken is not set properly. Set it to your Huggingface API token. Using default setting.")

    # Check Summerizeation Model
    try:
        summerizeationModel: str = settingsJson['summerizeationModel']
        
        if not summerizeationModel:
            warnings.warn("summerizeationModel is not set.")
        else:
            settingsJsonDict['summerizeationModel'] = summerizeationModel
    except KeyError:
        warnings.warn("KeyError: summerizeationModel is not set properly.")

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