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
    'summerizeationPrompt': 'You are a helpful assistant that summerizes text.'
}
