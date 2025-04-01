import json
from unittest.mock import patch, MagicMock
from ytscript.helpers import (
    getTimeString,
    getVideoLength,
    getVideoFilename,
    validateHuggingfaceToken,
    validateHuggingfaceModel,
    settingsJsonToDict,
    checkForTrueOrFalse,
    defineArguments
)
from ytscript.constants import DEFAULT_SETTINGS, SETTINGS_JSON_FILEPATH

class TestHelperFunctions:
    """Tests for helper functions in ytscript.helpers."""
    
    def test_getTimeString(self):
        """Test the getTimeString function formatting time values correctly."""
        # Test seconds only
        assert getTimeString(30) == "00:00:30"
        # Test minutes and seconds
        assert getTimeString(90) == "00:01:30"
        # Test hours, minutes, and seconds
        assert getTimeString(3661) == "01:01:01"
        # Test with float input
        assert getTimeString(30.5) == "00:00:30"
    
    @patch('ytscript.helpers.YoutubeDL')
    def test_getVideoLength(self, mockYoutubeDL):
        """Test getVideoLength correctly extracts duration from YouTube videos."""
        # Setup mock
        mockInstance = MagicMock()
        mockYoutubeDL.return_value.__enter__.return_value = mockInstance
        mockInstance.extract_info.return_value = {'duration': 300}
        
        # Test
        duration = getVideoLength("https://youtube.com/watch?v=test")
        
        # Assert
        assert duration == 300
        mockInstance.extract_info.assert_called_once_with("https://youtube.com/watch?v=test", download=False)
    
    @patch('ytscript.helpers.YoutubeDL')
    def test_getVideoFilename(self, mockYoutubeDL):
        """Test getVideoFilename correctly extracts title from YouTube videos."""
        # Setup mock
        mockInstance = MagicMock()
        mockYoutubeDL.return_value.__enter__.return_value = mockInstance
        mockInstance.extract_info.return_value = {'title': 'Test Video Title'}
        
        # Test
        filename = getVideoFilename("https://youtube.com/watch?v=test")
        
        # Assert
        assert filename == 'Test Video Title'
        mockInstance.extract_info.assert_called_once_with("https://youtube.com/watch?v=test", download=False)
    
    @patch('ytscript.helpers.HfApi')
    def test_validateHuggingfaceToken_valid(self, mockHfApi):
        """Test validateHuggingfaceToken returns True for valid tokens."""
        # Setup mock
        mockApi = MagicMock()
        mockHfApi.return_value = mockApi
        
        # Test
        result = validateHuggingfaceToken("valid_token")
        
        # Assert
        assert result is True
        mockHfApi.assert_called_once_with(token="valid_token")
    
    @patch('ytscript.helpers.HfApi')
    def test_validateHuggingfaceToken_invalid(self, mockHfApi):
        """Test validateHuggingfaceToken returns False for invalid tokens."""
        # Setup mock to raise exception
        mockHfApi.side_effect = Exception("Invalid token")
        
        # Test
        result = validateHuggingfaceToken("invalid_token")
        
        # Assert
        assert result is False
    
    @patch('ytscript.helpers.HfApi')
    def test_validateHuggingfaceModel_valid(self, mockHfApi):
        """Test validateHuggingfaceModel returns True for valid models."""
        # Setup mock
        mockApi = MagicMock()
        mockHfApi.return_value = mockApi
        
        # Test
        result = validateHuggingfaceModel("valid_model", "valid_token")
        
        # Assert
        assert result is True
        mockHfApi.assert_called_once_with(token="valid_token")
        mockApi.model_info.assert_called_once_with("valid_model")
    
    @patch('ytscript.helpers.HfApi')
    def test_validateHuggingfaceModel_invalid(self, mockHfApi):
        """Test validateHuggingfaceModel returns False for invalid models."""
        # Setup mock
        mockApi = MagicMock()
        mockHfApi.return_value = mockApi
        mockApi.model_info.side_effect = Exception("Model not found")
        
        # Test
        result = validateHuggingfaceModel("invalid_model", "valid_token")
        
        # Assert
        assert result is False
    
    def test_checkForTrueOrFalse(self):
        """Test checkForTrueOrFalse correctly identifies boolean string values."""
        # Test true values
        assert checkForTrueOrFalse("true") is True
        assert checkForTrueOrFalse("True") is True
        assert checkForTrueOrFalse("t") is True
        assert checkForTrueOrFalse("yes") is True
        assert checkForTrueOrFalse("Y") is True
        
        # Test false values
        assert checkForTrueOrFalse("false") is False
        assert checkForTrueOrFalse("False") is False
        assert checkForTrueOrFalse("f") is False
        assert checkForTrueOrFalse("no") is False
        assert checkForTrueOrFalse("N") is False
        
        # Test invalid values
        assert checkForTrueOrFalse("maybe") is None
        assert checkForTrueOrFalse("") is None
    
    @patch('ytscript.helpers.os.path.exists')
    @patch('ytscript.helpers.open')
    @patch('ytscript.helpers.json.load')
    @patch('ytscript.helpers.console.print')
    def test_settingsJsonToDict_file_not_found(self, mockPrint, mockJsonLoad, mockOpen, mockExists):
        """Test settingsJsonToDict returns default settings when file not found."""
        # Setup mock
        mockExists.return_value = False
        
        # Test
        result = settingsJsonToDict()
        
        # Assert
        assert result == DEFAULT_SETTINGS
    
    @patch('ytscript.helpers.os.path.exists')
    @patch('ytscript.helpers.open')
    @patch('ytscript.helpers.json.load')
    @patch('ytscript.helpers.console.print')
    def test_settingsJsonToDict_invalid_json(self, mockPrint, mockJsonLoad, mockOpen, mockExists):
        """Test settingsJsonToDict returns default settings when JSON is invalid."""
        # Setup mock
        mockExists.return_value = True
        mockJsonLoad.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Test
        result = settingsJsonToDict()
        
        # Assert
        assert result == DEFAULT_SETTINGS
    
    @patch('ytscript.helpers.os.path.exists')
    @patch('ytscript.helpers.open')
    @patch('ytscript.helpers.json.load')
    @patch('ytscript.helpers.console.print')
    def test_settingsJsonToDict_valid_settings(self, mockPrint, mockJsonLoad, mockOpen, mockExists):
        """Test settingsJsonToDict correctly loads valid settings."""
        # Setup mock
        mockExists.return_value = True
        valid_settings = {
            'outputFilepath': '/valid/path',
            'model': 'medium',
            'keepMp3': 'true',
            'summerize': 'true',
            'summerizeationModelType': 'local',
            'huggingfaceToken': 'valid_token',
            'summerizeationModel': 'valid_model',
            'summerizeationPrompt': 'valid_prompt'
        }
        mockJsonLoad.return_value = valid_settings
        mockExists.side_effect = lambda path: path == SETTINGS_JSON_FILEPATH or path == '/valid/path'
        
        # Test
        result = settingsJsonToDict()
        
        # Assert expected values from the valid settings
        assert result['outputFilepath'] == '/valid/path'
        assert result['model'] == 'medium'
        assert result['keepMp3'] == 'true'
    
    def test_defineArguments(self):
        """Test defineArguments sets up the argparse correctly."""
        with patch('sys.argv', ['script.py', 'https://youtube.com/watch?v=test']):
            args = defineArguments()
            assert args.videoUrl == 'https://youtube.com/watch?v=test'
            assert args.filepath is None
            assert args.keepMp3 is None
            assert args.model is None
