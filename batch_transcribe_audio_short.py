"""
Read in a manifest file and get all the required audio filepaths, then do a post request to whisper zero and do the transcription
"""

import os
import requests
import time
import json
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv

class BatchTranscribeAudio:

    '''
    The class to do the audio transcription
    '''

    def __init__(
        self,
        audio_root_path: str,
        input_manifest_path: str,
        output_manifest_path: str,
        language: str, 
    ) -> None:
        
        """
        audio_root_path: the root path of where the audio files reside
        input_manifest_path: manifest file to obtain the filepath of the audio files, nemo format
        output_manifest_path: raw manifest output from whisper zero
        language: target language of the audio clips 
        """

        self.audio_root_path = audio_root_path
        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path
        self.language = language

        load_dotenv()
        self.headers = {
            'x-gladia-key': os.environ.get("API_KEY"),
        }


    def load_manifest_nemo(self, input_manifest_path: str) -> List[Dict[str, str]]:

        '''
        loads the manifest file in Nvidia NeMo format to process the entries and store them into a list of dictionaries

        the manifest file would contain entries in this format:

        {"audio_filepath": "subdir1/xxx1.wav", "duration": 3.0, "text": "shan jie is an orange cat"}
        {"audio_filepath": "subdir1/xxx2.wav", "duration": 4.0, "text": "shan jie's orange cat is chonky"}
        ---

        input_manifest_path: the manifest path that contains the information of the audio clips of interest
        ---
        returns: a list of dictionaries of the information in the input manifest file
        '''

        dict_list = []

        with open(input_manifest_path, 'rb') as f:
            for line in f:
                dict_list.append(json.loads(line))

        return dict_list


    def transcribe_audio(self, input_audio_path: str) -> Dict:

        """
        method to transcribe a single audio file
        """

        with open(input_audio_path, 'rb') as f:
            files = {
                'audio': (input_audio_path, f, 'audio/wav'),
                'toggle_diarization': (None, True),
                'language_behaviour': 'manual',
                'language': self.language,
            }

            response = requests.post('https://api.gladia.io/audio/text/audio-transcription/', headers=self.headers, files=files)
            time.sleep(1.0)
            
            print(response.json())
            return response.json()
        

    def batch_transcribe_audio(self) -> None:

        """
        batch transcribe the audio files from manifest
        """

        # read the nemo json file
        manifest_list = self.load_manifest_nemo(input_manifest_path=self.input_manifest_path)

        output_json_list = []

        for entry in tqdm(manifest_list):
            response = self.transcribe_audio(input_audio_path=os.path.join(self.audio_root_path, entry['audio_filepath']))
            response['audio_filepath'] = entry['audio_filepath']

            output_json_list.append(response)

        # export file
        with open(self.output_manifest_path, "w") as f:
            json.dump(output_json_list, f, indent=2)


    def __call__(self):
        return self.batch_transcribe_audio()

if __name__ == '__main__':

    ROOT = '/datasets/mms/transcribed/mms_transcribed_batch_2/test_split'

    INPUT_MANIFEST = 'test_manifest_495.json'
    OUTPUT_MANIFEST = 'test_manifest_495_output_whisper_zero.json'

    b = BatchTranscribeAudio(
        audio_root_path=ROOT,
        input_manifest_path=os.path.join(ROOT, INPUT_MANIFEST),
        output_manifest_path=os.path.join(ROOT, OUTPUT_MANIFEST),
        language="english", 
    )()
