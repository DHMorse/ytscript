import os

SETTINGS_JSON_FILEPATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

DEFAULT_SETTINGS: dict[str, str] = {
    'outputFilepath': './',
    'model': 'tiny',
    'keepMp3': 'false',
    'summerize': 'false',
    'summerizeationModelType': 'local',
    'huggingfaceToken': 'YOUR_HUGGINGFACE_TOKEN',
    'summerizeationModel': 'meta-llama/Llama-3.2-3B-Instruct',
    "summerizeationPrompt": "Summarize the following YouTube video script in a way that is clear, concise, and engaging. Capture the key themes, main points, and essential takeaways while removing filler words and redundancy. Ensure the summary preserves the intent, tone, and emotional impact of the original content. If the script conveys a story, lesson, or argument, present it in a structured and digestible way. The summary should be direct, informative, and maintain the energy of the original message."
}
