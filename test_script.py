import os
import requests
from dotenv import load_dotenv

# load the environment variable
load_dotenv()

API_KEY = os.environ.get("API_KEY")
AUDIO_FILEPATH = '/datasets/long_1.wav'

headers = {
    'x-gladia-key': API_KEY,
}

files = {
    'audio_url': (None, AUDIO_FILEPATH),
    # 'audio_url': (None, 'http://files.gladia.io/example/audio-transcription/split_infinity.wav'),
    'toggle_diarization': (None, 'false'),
}

response = requests.post('https://api.gladia.io/audio/text/audio-transcription/', headers=headers, files=files)
print(response.json())