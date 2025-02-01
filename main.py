from yt_dlp import YoutubeDL
import whisper
import torch
import warnings
import sys
from typing import Dict, List
import webbrowser
import os
import json
import argparse

def downloadAudio(url: str, filepath: str) -> str:
    ydlOpts: Dict[str, str | List[Dict[str, str]]] = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{filepath}%(title)s.%(ext)s',
    }
    
    with YoutubeDL(ydlOpts) as ydl:
        ydl.download([url])

    filepath = ydl.prepare_filename(ydl.extract_info(url, download=False))
    return os.path.splitext(os.path.basename(filepath))[0]

def transcribeAudio(filepath: str, modelSize: str) -> str:
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model(modelSize, device=device)

    results = whisper.transcribe(model, filepath)

    return results['text']

def checkForTrueOrFalse(value: str) -> bool:
    if value.lower() in ['true', 't', 'yes', 'y']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n']:
        return False
    else:
        return False

def checkSettingsFile():
    if not os.path.exists('./settings.json'):
        print("settings.json not found. Creating a new one.")
        with open('./settings.json', 'w') as file:
            file.write('''{
    "outputFilepath": "./",
    "model": "large",
    "summerize": "false",
    "keepMp3": "true"
}''')

def defineArguments() -> argparse.Namespace:
    # Create argument parser
    parser = argparse.ArgumentParser(description="Demo for keyword arguments in CLI")

    # Define keyword arguments with -- prefix
    parser.add_argument("videoUrl", type=str, help="The URL of the video you want to transcribe")
    parser.add_argument("--filepath", type=str, required=False, help="The output file path of the mp3 and txt files")
    parser.add_argument("--model", type=str, required=False, help="The model size you want to use, either 'tiny, 'base', 'small', 'medium', or 'large'")
    parser.add_argument("--summerize", type=str, required=False, help="Whether to open the OpenAI chat to summerize the text")
    parser.add_argument("--keepMp3", type=str, required=False, help="Whether to keep the mp3 file after transcribing")

    # Parse arguments
    return parser.parse_args()

def main():
    if len(sys.argv) < 1:
        print("Usage: python script.py <video_url> [file_path] [model_size] [summerize (t/f/y/n)] [keep_mp3 (t/f/y/n)]")
        sys.exit(1)
    
    checkSettingsFile()

    args = defineArguments()

    with open('./settings.json', 'r') as file:
        settings = file.read()
        
        videoUrl: str = args.videoUrl
        filepath: str = args.filepath if args.filepath else json.loads(settings)['outputFilepath']
        modelSize: str = args.model if args.model else json.loads(settings)['model']
        summerize: str = args.summerize if args.summerize else json.loads(settings)['summerize']
        keepMp3: str = args.keepMp3 if args.keepMp3 else json.loads(settings)['keepMp3']

    summerize: bool = checkForTrueOrFalse(summerize)
    keepMp3: bool = checkForTrueOrFalse(keepMp3)

    filename: str = downloadAudio(videoUrl, filepath)
    transcribedText: str = transcribeAudio(filepath + f'{filename}.mp3', modelSize)

    with open(filepath + f'{filename}.txt', 'w') as file:
        file.write(transcribedText)

    if not keepMp3:
        os.remove(filepath + f'{filename}.mp3')

    if summerize:
        prompt: str = transcribedText.replace(' ', '%20')
        webbrowser.open(f'https://chat.openai.com/?q=summerize%20the%20text%20below%20{prompt}')

if __name__ == "__main__":
    main()