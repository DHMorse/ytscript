# Configuration Guide

## Settings File Location
The configuration file should be placed at `ytscript/settings.json`.

## Configuration Options

```json
{
    "outputFilepath": "",      // Directory path for output files (use '/' for Unix or '\\' for Windows)
    "model": "",              // Whisper model size: "tiny", "base", "small", "medium", "large"
    "keepMp3": false,         // Set to true to retain the MP3 file after transcription
    "summarize": false,       // Set to true to enable transcription summarization
    "summarizationModelType": "", // Summarization model type: "local" or "huggingface"
    "huggingfaceToken": "",   // Your Hugging Face API token (required for huggingface models)
    "summarizationModel": "", // Name/path of the summarization model
    "summarizationPrompt": "" // Custom prompt template for summarization
}
```