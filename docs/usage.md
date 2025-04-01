# Usage

Basic usage:
```
python ytscript/main.py <video_url>
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `videoUrl` | The URL of the YouTube video to transcribe (required) |
| `--filepath` | Output directory for MP3 and text files |
| `--keepMp3` | Whether to keep the MP3 file after transcription (`true`/`false`) |
| `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--summerize` | Whether to summarize the transcription (`true`/`false`) |
| `--summerizeationModelType` | Type of summarization model (`local` or `huggingface`) |
| `--huggingfaceToken` | Token for Hugging Face model access |
| `--summerizeationModel` | Model name to use for summarization |
| `--summerizeationPrompt` | Prompt to guide the summarization |

The size of the whisper model is directly related to the quality of the transcription; but I have to say that the quality of the tiny model is more than enough for most cases.

I recommend using meta-llama/Llama-3.2-3B-Instruct for a local summerization model.

For a huggingface model I recommend using Qwen/Qwen2.5-72B-Instruct.

Do note that you can most likely get a much better result by using a more powerful model on huggingface. The down side of course being that it isn't all local.