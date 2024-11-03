"""
Combine manifest to append the predictions into the raw nemo manifest 
"""

import os
import json
from tqdm import tqdm
from typing import Dict, List

from text_processing import TextPostProcessingManager

class CombineManifest:

    '''
    combine the manifest
    '''

    def __init__(
        self, 
        raw_manifest: str,
        whisper_zero_manifest: str,
        output_manifest: str,
        language: str,
    ) -> None:
        
        self.raw_manifest = raw_manifest
        self.whisper_zero_manifest = whisper_zero_manifest
        self.output_manifest = output_manifest
        self.language = language

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
    
    def load_whisper_zero_manifest(self, input_manifest_path: str) -> Dict[str, str]:

        '''
        load the whisper zero manifest and then concat the pred transcriptions together
        {
            <filepath>: transcription,
            ...
        }
        '''

        transcription_dict = {}

        with open(input_manifest_path, 'rb') as f:
            data = json.load(f)

        for entry in tqdm(data):
            temp_transcription_list = []

            for pred in entry['prediction']:
                temp_transcription_list.append(pred['transcription'])

            final_transciption = ' '.join(temp_transcription_list)
            final_transciption_cleaned = TextPostProcessingManager(
                language=self.language
            ).process_data(text=final_transciption)
            transcription_dict[entry['audio_filepath']] = {
                'pred_str_raw': final_transciption,
                'pred_str': final_transciption_cleaned
            }

        return transcription_dict
    
    def combine_manifest(self) -> None:

        # load the manifests
        manifest_nemo = self.load_manifest_nemo(input_manifest_path=self.raw_manifest)
        pred_transcription_dict = self.load_whisper_zero_manifest(input_manifest_path=self.whisper_zero_manifest)

        for entry in tqdm(manifest_nemo):
            entry['text_raw'] = entry['text']
            entry['text'] = TextPostProcessingManager(
                language=self.language
            ).process_data(text=entry['text'])
            entry['pred_str'] = pred_transcription_dict[entry['audio_filepath']]['pred_str']
            entry['pred_str_raw'] = pred_transcription_dict[entry['audio_filepath']]['pred_str_raw']

        # export the manifest file
        with open(self.output_manifest, 'w+', encoding='utf-8') as f:
            for data in manifest_nemo:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def __call__(self) -> None:
        return self.combine_manifest()


if __name__ == '__main__':

    ROOT = '/datasets/mms/transcribed/mms_transcribed_batch_2/test_split'

    RAW_MANIFEST = 'test_manifest_495.json'
    WHISPER_ZERO_MANIFEST = 'test_manifest_495_output_whisper_zero.json'
    OUTPUT_MANIFEST = 'test_manifest_495_with_pred.json'

    c = CombineManifest(
        raw_manifest=os.path.join(ROOT, RAW_MANIFEST),
        whisper_zero_manifest=os.path.join(ROOT, WHISPER_ZERO_MANIFEST),
        output_manifest=os.path.join(ROOT, OUTPUT_MANIFEST),
        language='en'
    )()