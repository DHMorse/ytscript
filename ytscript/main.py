import time
libraryImportStartTime = time.time()

from transformers.generation.streamers import TextIteratorStreamer # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from huggingface_hub import login as huggingfaceLogin # type: ignore
from yt_dlp import YoutubeDL # type: ignore
import whisper # type: ignore
import argparse
import warnings
import sys
import os
import torch
from tqdm import tqdm
from threading import Thread

from helpers import settingsJsonToDict, checkForTrueOrFalse, defineArguments

libraryImportEndTime = time.time()
print(f"Library import time: {round(libraryImportEndTime - libraryImportStartTime, 2)} seconds")

def getVideoFilename(url: str) -> str:
    """
    Extracts the video filename from a YouTube URL.

    Args:
        url (str): The URL of the YouTube video.
    """
    with YoutubeDL() as ytDL:
        infoDict: dict = ytDL.extract_info(url, download=False)
        videoTitle: str = infoDict.get('title', '')
        return videoTitle
    
def downloadVideoAudio(url: str, filepath: str) -> None:
    """
    Downloads the audio track from a YouTube video URL and returns the base filename.

    Args:
        url (str): The URL of the YouTube video.
        filepath (str): The directory path where the audio file should be saved.

    Returns:
        str: The base filename (without extension) of the downloaded audio file.
    """
    ytDLoptions: dict[str, str | list[dict[str, str]]] = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{filepath}%(title)s.%(ext)s',
    }
    
    with YoutubeDL(ytDLoptions) as ytDL:
        infoDict: dict = ytDL.extract_info(url, download=False)
        # Ensure download happens before getting filename if not already present
        if not os.path.exists(ytDL.prepare_filename(infoDict)):
             ytDL.download([url])

def transcribeAudio(filepath: str, modelSize: str) -> str:
    """
    Transcribes audio from a file using the Whisper model.
    
    Args:
        filepath (str): Path to the audio file to transcribe
        modelSize (str): Size of the Whisper model to use
        
    Returns:
        str: The transcribed text
    """
    # Suppress warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.cuda.empty_cache()

    if device == "cpu":
        warnings.warn("Using CPU for transcription, because CUDA is not available")
    

    print("Loading Whisper model...")
    model: whisper.Whisper = whisper.load_model(modelSize, device=device)
        
    print("Transcribing audio... (This may take a while)")
    results: whisper.transcribe.Result = whisper.transcribe(model, filepath)

    return results['text']

def summerizeTextLocal(text: str, modelName: str, prompt: str) -> str:
    """
    Summarizes text using a local language model.
    
    Args:
        text (str): The text to summarize.
        modelName (str): The model name or path to use for summarization.
                        Default is microsoft/Phi-4-mini-instruct.
        prompt (str): The system prompt to guide the summarization.
        
    Returns:
        str: The summarized text.
    """
    
    # Check for GPU availability
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    if device == "cpu":
        warnings.warn("Using CPU for summarization, which may be slow. CUDA is not available.")
    
    print(f"Loading model {modelName}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    
    # Format messages with system prompt and user text for instruct models
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    
    # Format conversation for the tokenizer
    inputText = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer(inputText, return_tensors="pt").to(device)
    
    # Set max tokens
    maxTokens: int = 4096
    
    # Set up streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Create improved generation config for better summaries
    generationConfig = {
        "max_new_tokens": maxTokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.5,  # Lower temperature for more focused output
        "top_p": 0.95,
        "repetition_penalty": 1.1,  # Slight penalty for repetitive text
        "streamer": streamer
    }
    
    # Start generation in a separate thread
    generationThread = Thread(
        target=lambda: model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generationConfig
        )
    )
    
    print("Generating summary...")
    generationThread.start()
    
    # Collect the generated text and update progress bar
    generatedText: str = ""
    progressBar = tqdm(total=maxTokens, desc="Summarizing")
    
    for textChunk in streamer:
        generatedText += textChunk
        # Update progress based on tokens generated so far
        progressBar.update(1)
    
    # Ensure progress bar reaches 100% when done
    progressBar.n = maxTokens
    progressBar.refresh()
    progressBar.close()
    
    # Clean up any artifacts in the output
    cleanedText: str = generatedText.strip()
    
    return cleanedText.strip()

def summerizeTextHuggingface(text: str, model: str, prompt: str, token: str) -> str:
    """
    Placeholder for summarizing text using a Hugging Face model via API or inference endpoint.
    (Currently not implemented)

    Args:
        text (str): The text to summarize.
        model (str): The Hugging Face model identifier.
        prompt (str): The prompt for summarization.
        token (str): Hugging Face API token.

    Returns:
        str: An empty string (as it's not implemented).
    """
    return ''

def main() -> None:
    if len(sys.argv) <= 1:
        print("Basic Usage: python main.py <video_url>")
        sys.exit(1)

    settingsJson: dict[str, str] = settingsJsonToDict()

    args: argparse.Namespace = defineArguments()

    videoUrl: str = args.videoUrl

    filepathArgument: str | None = args.filepath
    if filepathArgument is not None and not os.path.exists(filepathArgument):
        raise FileNotFoundError(f"The filepath {filepathArgument} does not exist")
    elif filepathArgument is None:
        filepath: str = settingsJson['outputFilepath']

    modelSizeArgument: str | None = args.model
    if modelSizeArgument is not None and modelSizeArgument not in ["tiny", "base", "small", "medium", "large"]:
        raise ValueError(f"The model size {modelSizeArgument} is not valid. Please use one of the following: tiny, base, small, medium, large")
    elif modelSizeArgument is None:
        modelSize: str = settingsJson['model']

    keepMp3Argument: str | None = args.keepMp3
    if keepMp3Argument is not None and checkForTrueOrFalse(keepMp3Argument) is None:
        raise ValueError(f"The keepMp3 argument {keepMp3Argument} is not valid. Please use one of the following: true, false")
    elif keepMp3Argument is None:
        keepMp3: str = settingsJson['keepMp3']

    shouldKeepMp3: bool | None = checkForTrueOrFalse(keepMp3)

    summerizeArgument: str | None = args.summerize
    if summerizeArgument is not None and checkForTrueOrFalse(summerizeArgument) is None:
        raise ValueError(f"The summerize argument {summerizeArgument} is not valid. Please use one of the following: true, false")
    elif summerizeArgument is None:
        summerize: str = settingsJson['summerize']

    shouldSummerize: bool | None = checkForTrueOrFalse(summerize)

    if shouldSummerize:
        summerizeationModelTypeArgument: str | None = args.summerizeationModelType
        if summerizeationModelTypeArgument is not None and summerizeationModelTypeArgument not in ['local', 'huggingface']:
            raise ValueError(f"The summerizeationModelType argument {summerizeationModelTypeArgument} is not valid. Please use one of the following: local, huggingface")
        elif summerizeationModelTypeArgument is None:
            summerizeationModelType: str = settingsJson['summerizeationModelType']

        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        huggingfaceTokenArgument: str | None = args.huggingfaceToken
        if huggingfaceTokenArgument is not None:
            try:
                huggingfaceLogin(huggingfaceTokenArgument)
            except Exception as e:
                raise ValueError(f"The huggingfaceToken argument {huggingfaceTokenArgument} is not valid. Please use a valid huggingface token.")
        elif huggingfaceTokenArgument is None:
            huggingfaceToken: str = settingsJson['huggingfaceToken']

        summerizeationModelArgument: str | None = args.summerizeationModel
        if summerizeationModelArgument is not None:
            if summerizeationModelType == 'huggingface':
                try:
                    huggingfaceLogin(huggingfaceToken)
                except Exception as e:
                    raise ValueError(f"The summerizeationModel argument {summerizeationModelArgument} is not valid. Please use a valid huggingface model.")
            elif summerizeationModelType == 'local':
                if not os.path.exists(summerizeationModelArgument):
                    pass
                # to do: add validation for local model
        else:
            summerizeationModel: str = settingsJson['summerizeationModel']
        
        summerizeationPromptArgument: str | None = args.summerizeationPrompt
        if summerizeationPromptArgument is not None:
            summerizeationPrompt: str = summerizeationPromptArgument
        elif summerizeationPromptArgument is None:
            summerizeationPrompt = settingsJson['summerizeationPrompt']

    videoTitle: str = getVideoFilename(videoUrl)
    print(f"Video title: `{videoTitle}`")
    videoFilepath: str = filepath + f'{videoTitle}.mp3'
    transcribedTextFilepath: str = filepath + f'{videoTitle}.txt'
    summerizedTextFilepath: str = filepath + f'summerized_{videoTitle}.txt'

    if not os.path.exists(transcribedTextFilepath):
        downloadVideoAudio(videoUrl, filepath)

    # add a progress bar for the transcribeAudio function
        transcribedText: str = transcribeAudio(filepath + f'{videoTitle}.mp3', modelSize)

        with open(transcribedTextFilepath, 'w') as file:
            file.write(transcribedText)

    else:
        print(f"Transcribed text already exists for `{transcribedTextFilepath}`. Reading from file...")
        with open(transcribedTextFilepath, 'r') as file:
            transcribedText = file.read()

    if not shouldKeepMp3:
        if os.path.exists(videoFilepath):
            os.remove(videoFilepath)

    if shouldSummerize:
        if summerizeationModelType == 'local':
            summerizedText: str = summerizeTextLocal(transcribedText, summerizeationModel, summerizeationPrompt)
        elif summerizeationModelType == 'huggingface':
            summerizedText = summerizeTextHuggingface(transcribedText, summerizeationModel, summerizeationPrompt, huggingfaceToken)

        with open(summerizedTextFilepath, 'w') as file:
            file.write(summerizedText)

if __name__ == "__main__":
    main()