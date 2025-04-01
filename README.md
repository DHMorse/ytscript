# YTScript 🎥 → 📝

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-FFE873?style=flat&logo=python&logoColor=black)](https://github.com/astral-sh/uv)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-FF0000?style=flat&logo=youtube&logoColor=white)](https://github.com/yt-dlp/yt-dlp)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-white?style=flat&logo=openai&logoColor=black)](https://github.com/openai/whisper)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)


A powerful Python tool that transcribes and summarizes YouTube videos using OpenAI's Whisper for transcription and state-of-the-art language models for summarization.

## Features

- 🎯 **Accurate Transcription**: Uses OpenAI's Whisper model for high-quality speech-to-text conversion
- 📝 **Smart Summarization**: Generates concise, meaningful summaries using advanced language models
- 🌐 **Flexible Model Options**: Choose between local models or Hugging Face's API
- 🚀 **Easy to Use**: Simple command-line interface with sensible defaults
- 💾 **Resource Efficient**: Optimized for both high-end and resource-constrained systems

## System Requirements

- Operating System: Linux, macOS, or Windows
- Python 3.10 or higher (if building from source)
- Internet connection (for model downloads and Hugging Face API)

## Installation

### Method 1: Pre-built Binary (Recommended)

1. Download the appropriate binary for your system from the [releases page](https://github.com/DHMorse/ytscript/releases)
2. Make the file executable (Unix systems):
   ```bash
   chmod +x ytscript
   ```
3. Move to a directory in your PATH:
   ```bash
   sudo mv ytscript /usr/local/bin/
   ```

### Method 2: Build from Source

See our detailed [build instructions](docs/build-from-source.md) for compiling from source code.

## Documentation

- [Usage Guide](docs/usage.md) - Learn about available commands and options
- [Configuration Guide](docs/configuration.md) - Customize YTScript for your needs

## Example Usage

```bash
# Basic transcription
ytscript "https://youtu.be/example" --model tiny

# Transcribe and summarize using a local model
ytscript "https://youtu.be/example" --summarize true

# Use Hugging Face for summarization
ytscript "https://youtu.be/example" --summarize true --summarizationModelType huggingface
```

## Development

### Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_transcription.py -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 Check our [documentation](docs/) for detailed information
- 🐛 Report bugs in our [issue tracker](https://github.com/DHMorse/ytscript/issues)
- 💡 Request features in our [discussions](https://github.com/DHMorse/ytscript/discussions)