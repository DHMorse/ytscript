# Configuration

Configuration can be set in `ytscript/settings.json`:

```json
{
    "outputFilepath": "", // The path to save the output file this can be any path you want
    "model": "", // The model to use for transcription (tiny, base, small, medium, large)
    "keepMp3": "", // Whether to keep the MP3 file after transcription (true, false)
    "summerize": "", // Whether to summarize the transcription (true, false)
    "summerizeationModelType": "", // The type of summarization model to use (local, huggingface)
    "huggingfaceToken": "", // The token for Hugging Face model access
    "summerizeationModel": "", // The model to use for summarization
    "summerizeationPrompt": "" // The prompt to guide the summarization
}
```