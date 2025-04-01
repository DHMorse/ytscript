import time
libraryImportStartTime = time.time()

from transformers.generation.streamers import TextIteratorStreamer # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from huggingface_hub import login as huggingfaceLogin 
from huggingface_hub import InferenceClient
from yt_dlp import YoutubeDL # type: ignore
import whisper # type: ignore
import argparse
import warnings
import sys
import os
import torch
from tqdm import tqdm
from threading import Thread
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live

from ytscript.helpers import ( 
    richWarning,
    settingsJsonToDict, 
    checkForTrueOrFalse, 
    defineArguments, 
    getVideoLength, 
    getVideoFilename, 
    getTimeString, 
    validateHuggingfaceToken, 
    validateHuggingfaceModel
)

console = Console()

warnings.formatwarning = richWarning

libraryImportEndTime = time.time()
console.print(f"Library import time: [blue]{round(libraryImportEndTime - libraryImportStartTime, 2)} seconds[/]")

def downloadVideoAudio(url: str, filepath: str) -> None:
    """
    Downloads the audio track from a YouTube video URL

    Args:
        url (str): The URL of the YouTube video.
        filepath (str): The directory path where the audio file should be saved.
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
    

    panel = Panel("Loading Whisper model...", title="[bold blue]Information[/]")
    console.print(panel)
    model: whisper.Whisper = whisper.load_model(modelSize, device=device)
        
    panel = Panel("Transcribing audio... (This may take a while)", title="[bold blue]Information[/]")
    console.print(panel)
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
        panel = Panel("Using CPU for summarization, which may be slow. CUDA is not available.", title="[bold red]Pretty Warning[/]")
        console.print(panel)
    
    panel = Panel(f"Loading model {modelName}...", title="[bold blue]Information[/]")
    console.print(panel)
    
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
    
    generatedText: str = ""
    
    # Create a Live display within a panel
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    
    panel = Panel(progress, title="[bold blue]Information[/]", border_style="white")
    
    with Live(panel, refresh_per_second=10, console=console):
        summaryTask = progress.add_task("[cyan]Generating summary...", total=maxTokens)
        
        # Start the generation
        generationThread.start()
        
        # Update progress as we receive chunks
        for textChunk in streamer:
            generatedText += textChunk
            progress.update(summaryTask, advance=1)
            
        # Ensure the progress reaches 100%
        progress.update(summaryTask, completed=maxTokens)
    
    # Clean up any artifacts in the output
    cleanedText: str = generatedText.strip()
    
    return cleanedText.strip()

def summerizeTextHuggingface(text: str, model: str, prompt: str, token: str) -> str:
    """
    Summarizes text using a Hugging Face model via API or inference endpoint.

    Args:
        text (str): The text to summarize.
        model (str): The Hugging Face model identifier.
        prompt (str): The prompt for summarization.
        token (str): Hugging Face API token.

    Returns:
        str: The summarized text.

    Raises:
        SystemExit: If there's a payment required error or other critical Hugging Face API errors
    """
    maxTokens: int = 4096
    
    panel = Panel(f"Using Hugging Face model {model} for summarization...", title="[bold blue]Information[/]")
    console.print(panel)
    
    # Authenticate with Hugging Face
    huggingfaceLogin(token)
    
    # Set up the inference client
    inferenceClient = InferenceClient(model=model, token=token)

    # Generate the summary
    try:
        summary = inferenceClient.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=maxTokens,
            temperature=0.5,
            top_p=0.95,
            stream=True
        )
    except Exception as e:
        errorMessage: str = str(e)
        if "402" in errorMessage and "Payment Required" in errorMessage:
            panel = Panel(
                "[red]You have exceeded your monthly included credits for Inference Providers.[/]\n"
                "[yellow]To resolve this, you can either:[/]\n"
                "1. Subscribe to Hugging Face Pro for more credits\n"
                "2. Use a different model\n"
                "3. Switch to local model inference",
                title="[bold red]Error: Payment Required[/]"
            )
        else:
            panel = Panel(
                f"[red]An error occurred while generating the summary:[/]\n{errorMessage}",
                title="[bold red]Error: Hugging Face API[/]"
            )
        console.print(panel)
        sys.exit(1)

    # Create a Live display within a panel
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    
    panel = Panel(progress, title="[bold blue]Information[/]", border_style="white")
    
    generatedText: str = ""
    
    with Live(panel, refresh_per_second=10, console=console):
        summaryTask = progress.add_task("[cyan]Generating summary...", total=maxTokens)
        
        for textChunk in summary:
            generatedText += textChunk.choices[0].delta.content
            progress.update(summaryTask, advance=1)
            
        # Ensure the progress reaches 100%
        progress.update(summaryTask, completed=maxTokens)
    
    # Clean up any artifacts in the output
    cleanedText: str = generatedText.strip()
    
    return cleanedText.strip()


def main() -> None:
    if len(sys.argv) <= 1:
        panel = Panel("Basic Usage: python main.py <video_url>", title="[bold red]Error[/]")
        console.print(panel)
        sys.exit(1)

    startTime: float = time.time()

    settingsJson: dict[str, str] = settingsJsonToDict()
    args: argparse.Namespace = defineArguments()

    videoUrl: str = args.videoUrl
    videoLength: int = getVideoLength(videoUrl)

    # Initialize variables with defaults to avoid scope issues
    filepath: str = settingsJson['outputFilepath']
    modelSize: str = settingsJson['model']
    keepMp3: str = settingsJson['keepMp3']
    summerize: str = settingsJson['summerize']
    summerizeationModelType: str = settingsJson['summerizeationModelType']
    summerizeationModel: str = settingsJson['summerizeationModel']
    summerizeationPrompt: str = settingsJson['summerizeationPrompt']
    huggingfaceToken: str = settingsJson['huggingfaceToken']



    # Handle filepath argument
    filepathArgument: str | None = args.filepath
    if filepathArgument is not None and not os.path.exists(filepathArgument):
        raise FileNotFoundError(f"The filepath {filepathArgument} does not exist")
    elif filepathArgument is not None:
        filepath = filepathArgument



    # Handle modelSize argument
    modelSizeArgument: str | None = args.model
    if modelSizeArgument is not None and modelSizeArgument not in ["tiny", "base", "small", "medium", "large"]:
        raise ValueError(f"The model size {modelSizeArgument} is not valid. Please use one of the following: tiny, base, small, medium, large")
    elif modelSizeArgument is not None:
        modelSize = modelSizeArgument



    # Handle keepMp3 argument
    keepMp3Argument: str | None = args.keepMp3
    if keepMp3Argument is not None and checkForTrueOrFalse(keepMp3Argument) is None:
        raise ValueError(f"The keepMp3 argument {keepMp3Argument} is not valid. Please use one of the following: true, false")
    elif keepMp3Argument is not None:
        keepMp3 = keepMp3Argument
    shouldKeepMp3: bool | None = checkForTrueOrFalse(keepMp3)



    # Handle summerize argument
    summerizeArgument: str | None = args.summerize
    if summerizeArgument is not None and checkForTrueOrFalse(summerizeArgument) is None:
        raise ValueError(f"The summerize argument {summerizeArgument} is not valid. Please use one of the following: true, false")
    elif summerizeArgument is not None:
        summerize = summerizeArgument
    shouldSummerize: bool | None = checkForTrueOrFalse(summerize)



    if shouldSummerize:
        # Handle summerizeationModelType argument
        summerizeationModelTypeArgument: str | None = args.summerizeationModelType
        if summerizeationModelTypeArgument is not None and summerizeationModelTypeArgument not in ['local', 'huggingface']:
            raise ValueError(f"The summerizeationModelType argument {summerizeationModelTypeArgument} is not valid. Please use one of the following: local, huggingface")
        elif summerizeationModelTypeArgument is not None:
            summerizeationModelType = summerizeationModelTypeArgument


        # Handle summerizeationModel argument
        summerizeationModelArgument: str | None = args.summerizeationModel
        if summerizeationModelArgument is not None:
            summerizeationModel = summerizeationModelArgument


        # Handle Model Validation
        if summerizeationModelType == 'local':
            # there is currently no validation for the local model
            pass

        elif summerizeationModelType == 'huggingface':
            # Validate hugging face token
            if not validateHuggingfaceToken(huggingfaceToken):
                raise ValueError(f"The huggingfaceToken argument {huggingfaceToken} is not valid. Please check your token and try again.")


            # Validate hugging face model
            if not validateHuggingfaceModel(summerizeationModel, huggingfaceToken):
                raise ValueError(f"The summerizeationModel argument {summerizeationModel} is not valid. Please check your model and try again.")



        # Handle summerizeationPrompt argument
        summerizeationPromptArgument: str | None = args.summerizeationPrompt
        if summerizeationPromptArgument is not None:
            summerizeationPrompt = summerizeationPromptArgument


    videoTitle: str = getVideoFilename(videoUrl)
    panel = Panel(f"Video title: `{videoTitle}`", title="[bold blue]Information[/]")
    console.print(panel)
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
        panel = Panel(f"Transcribed text already exists for `{transcribedTextFilepath}`. Reading from file...", title="[bold blue]Information[/]")
        console.print(panel)
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

    endTime: float = time.time()
    timeSpent: float = endTime - startTime
    timeSaved: float = videoLength - timeSpent


    if shouldSummerize and timeSaved < videoLength:
        panel = Panel(
            f"Video length: [green]{getTimeString(videoLength)}[/]\n"
            f"Time spent: [yellow]{getTimeString(timeSpent)}[/]\n"
            f"Time saved: [cyan]{getTimeString(timeSaved)}[/]",
            title="[bold blue]Information[/]"
        )
        console.print(panel)

    exit(0)

if __name__ == "__main__":
    main()