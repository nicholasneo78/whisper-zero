import os
import json
import requests
from dotenv import load_dotenv

# load the environment variable
load_dotenv()

API_KEY = os.environ.get("API_KEY")
AUDIO_FILEPATH = '/datasets/long_2_id.wav'
OUTPUT = 'output.json'

# FILENAME = 'CHDIR_497_2022-07-15'

# AUDIO_FILEPATH = f'/datasets/mms/transcribed/mms_transcribed_batch_2/test/{FILENAME}.wav'

# OUTPUT = f'/datasets/mms/transcribed/mms_transcribed_batch_2/test_split/{FILENAME}_whisper_zero.json'

headers = {
    'x-gladia-key': API_KEY,
}

with open(AUDIO_FILEPATH, 'rb') as f:
    files = {
        'audio': (AUDIO_FILEPATH, f, 'audio/wav'),
        # 'audio_url': (None, 'http://files.gladia.io/example/audio-transcription/split_infinity.wav'),
        'toggle_diarization': (None, True),
        'language_behaviour': 'manual',
        'language': 'indonesian'
    }

    response = requests.post('https://api.gladia.io/audio/text/audio-transcription/', headers=headers, files=files)
    print(response.json())

    with open(OUTPUT, "w") as f:
        json.dump(response.json(), f, indent=2)