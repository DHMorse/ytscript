# Usage Guide


## Command Line Arguments

| Argument | Description | Available Values | Default |
|----------|-------------|------------------|---------|
| `videoUrl` | YouTube video URL to transcribe (required) | Any valid YouTube URL | - |
| `--filepath` | Output directory for generated files | Any valid directory path | `./` (Current directory) |
| `--keepMp3` | Keep the MP3 file after transcription | `true`, `t`, `false`,  `f`, `yes`, `y`, `no`, `n` | `false` |
| `--model` | Whisper model size | `tiny`, `base`, `small`, `medium`, `large` | `tiny` |
| `--summerize` | Enable transcription summarization | `true`, `t`, `false`, `f`, `yes`, `y`, `no`, `n` | `false` |
| `--summerizeationModelType` | Summarization model type | `local`, `huggingface` | `local` |
| `--huggingfaceToken` | Hugging Face API token | Any valid Hugging Face token | `YOUR_HUGGINGFACE_TOKEN` |
| `--summerizeationModel` | Model name for summarization | Any compatible model name | `meta-llama/Llama-3.2-3B-Instruct` |
| `--summerizeationPrompt` | Custom prompt for summarization | Any text string | Default summarization prompt* |

*Default summarization prompt: "Summarize the following YouTube video script in a way that is clear, concise, and engaging. Capture the key themes, main points, and essential takeaways while removing filler words and redundancy. Ensure the summary preserves the intent, tone, and emotional impact of the original content. If the script conveys a story, lesson, or argument, present it in a structured and digestible way. The summary should be direct, informative, and maintain the energy of the original message."

## Model Recommendations

### Transcription Models
The Whisper model size directly impacts transcription quality. The `tiny` model provides sufficient accuracy for most use cases while being the most resource-efficient.

### Summarization Models
For summarization, we recommend:

- **Local**: 
  - Advantages: Privacy, no API costs, offline usage, less sensorship
  - Best for: Personal use on mid-ranged to high-end machines
  - Recommended model: `meta-llama/Llama-3.2-3B-Instruct`

- **Hugging Face**: 
  - Advantages: No local processing required, possibly higher quality
  - Best for: Personal use on low-end machines
  - Recommended model: `microsoft/phi-4`

Note: While Hugging Face models generally provide better results, they require an internet connection and API token.