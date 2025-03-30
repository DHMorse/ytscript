import time
libraryImportStartTime = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from huggingface_hub import login as huggingfaceLogin # type: ignore
from yt_dlp import YoutubeDL # type: ignore
import whisper
import argparse
import warnings
import sys
import os
import torch

from helpers import settingsJsonToDict, checkForTrueOrFalse, defineArguments

libraryImportEndTime = time.time()
print(f"Library import time: {libraryImportEndTime - libraryImportStartTime} seconds")

def downloadAudioAndGetFilename(url: str, filepath: str) -> str:
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
        ytDL.download([url])

    filepath = ytDL.prepare_filename(ytDL.extract_info(url, download=False))
    return os.path.splitext(os.path.basename(filepath))[0]

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
    warnings.filterwarnings("ignore", category=FutureWarning)
    
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

def summerizeTextLocal(text: str, model: str, prompt: str) -> str:
    """
    Summarizes text using a local language model with GPU acceleration.
    
    Args:
        text (str): The text to summarize
        model (str): The model identifier or path to use
        prompt (str): The prompt template to use for summarization
        
    Returns:
        str: The summarized text
        
    Raises:
        ValueError: If the model fails to load or generate text
        RuntimeError: If there's an error during text generation
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        warnings.warn("Using CPU for summarization, because CUDA is not available")

    try:
        print(f"Loading model {model}...")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
        languageModel: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Prepare the input text with the prompt template
        inputText: str = f"{prompt}\n\nText to summarize:\n{text}\n\nSummary:"
        inputs: dict[str, torch.Tensor] = tokenizer(inputText, return_tensors="pt").to(device)
        
        print("Generating summary...")
        outputs: torch.Tensor = languageModel.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        summary: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the summary part after the prompt
        summary = summary.split("Summary:")[-1].strip()
        
        return summary
        
    except Exception as e:
        raise ValueError(f"Failed to generate summary: {str(e)}")

def summerizeTextHuggingface(text: str, model: str, prompt: str, token: str) -> str:
    return ''

def main() -> None:
    if len(sys.argv) <= 1:
        print("Basic Usage: python main.py <video_url>")
        sys.exit(1)

    settingsJson: dict[str, str] = settingsJsonToDict()

    args: argparse.Namespace = defineArguments()

    videoUrl: str = args.videoUrl

    filepath: str | None = args.filepath
    if filepath is not None and not os.path.exists(filepath):
        raise FileNotFoundError(f"The filepath {filepath} does not exist")
    elif filepath is None:
        filepath: str = settingsJson['outputFilepath']

    modelSize: str | None = args.model
    if modelSize is not None and modelSize not in ["tiny", "base", "small", "medium", "large"]:
        raise ValueError(f"The model size {modelSize} is not valid. Please use one of the following: tiny, base, small, medium, large")
    elif modelSize is None:
        modelSize: str = settingsJson['model']

    keepMp3: str | None = args.keepMp3
    if keepMp3 is not None and checkForTrueOrFalse(keepMp3) is None:
        raise ValueError(f"The keepMp3 argument {keepMp3} is not valid. Please use one of the following: true, false")
    elif keepMp3 is None:
        keepMp3: str = settingsJson['keepMp3']

    shouldKeepMp3: bool | None = checkForTrueOrFalse(keepMp3)

    summerize: str | None = args.summerize
    if summerize is not None and checkForTrueOrFalse(summerize) is None:
        raise ValueError(f"The summerize argument {summerize} is not valid. Please use one of the following: true, false")
    elif summerize is None:
        summerize: str = settingsJson['summerize']

    shouldSummerize: bool | None = checkForTrueOrFalse(summerize)

    if shouldSummerize:
        summerizeationModelType: str | None = args.summerizeationModelType
        if summerizeationModelType is not None and summerizeationModelType not in ['local', 'huggingface']:
            raise ValueError(f"The summerizeationModelType argument {summerizeationModelType} is not valid. Please use one of the following: local, huggingface")
        elif summerizeationModelType is None:
            summerizeationModelType: str = settingsJson['summerizeationModelType']

        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        huggingfaceToken: str | None = args.huggingfaceToken
        if huggingfaceToken is not None:
            try:
                huggingfaceLogin(huggingfaceToken)
            except Exception as e:
                raise ValueError(f"The huggingfaceToken argument {huggingfaceToken} is not valid. Please use a valid huggingface token.")
        elif huggingfaceToken is None:
            huggingfaceToken: str = settingsJson['huggingfaceToken']
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN
        # ADD PROPER VALIDATION FOR HUGGINGFACE TOKEN

        summerizeationModel: str | None = args.summerizeationModel
        if summerizeationModel is not None:
            if summerizeationModelType == 'huggingface':
                try:
                    huggingfaceLogin(huggingfaceToken)
                except Exception as e:
                    raise ValueError(f"The summerizeationModel argument {summerizeationModel} is not valid. Please use a valid huggingface model.")
            elif summerizeationModelType == 'local':
                if not os.path.exists(summerizeationModel):
                    pass
                # to do: add validation for local model
        else:
            summerizeationModel: str = settingsJson['summerizeationModel']
        
        summerizeationPrompt: str | None = args.summerizeationPrompt
        if summerizeationPrompt is not None:
            summerizeationPrompt: str = summerizeationPrompt
        elif summerizeationPrompt is None:
            summerizeationPrompt: str = settingsJson['summerizeationPrompt']

    filename: str = downloadAudioAndGetFilename(videoUrl, filepath)

    # add a progress bar for the transcribeAudio function
    transcribedText: str = transcribeAudio(filepath + f'{filename}.mp3', modelSize)

    with open(filepath + f'{filename}.txt', 'w') as file:
        file.write(transcribedText)

    if not shouldKeepMp3:
        os.remove(filepath + f'{filename}.mp3')

    if shouldSummerize:
        if summerizeationModelType == 'local':
            summerizedText: str = summerizeTextLocal(transcribedText, summerizeationModel, summerizeationPrompt)
        elif summerizeationModelType == 'huggingface':
            summerizedText: str = summerizeTextHuggingface(transcribedText, summerizeationModel, summerizeationPrompt)

        with open(filepath + f'{filename}_summerized.txt', 'w') as file:
            file.write(summerizedText)

if __name__ == "__main__":
    main()