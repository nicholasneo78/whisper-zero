"""
Pull all the single word and the timestamps of the words
"""

import os
import json
from tqdm import tqdm
from typing import Dict, List

class ExtractSingleWord:

    '''
    extract single word
    '''

    def __init__(self, input_manifest: str, output_manifest: str) -> None:
        self.input_manifest = input_manifest
        self.output_manifest = output_manifest

    
    def extract(self) -> None:

        """
        main method to do the extraction
        """

        word_list = []

        with open(self.input_manifest , 'rb') as f:
            data = json.load(f)['prediction']

        for entry in tqdm(data):
            for word in entry['words']:
                temp = {
                    "text": word['word'],
                    "start": word['time_begin'],
                    "end": word['time_end'],
                    "confidence": word['confidence']
                }

                word_list.append(temp)
        
        # export manifest
        with open(self.output_manifest, 'w+', encoding='utf-8') as f:
            for word in word_list:
                f.write(json.dumps(word, ensure_ascii=False) + '\n')

    def __call__(self) -> None:
        self.extract()


if __name__ == '__main__':

    ROOT = '/datasets/mms/transcribed/mms_transcribed_batch_2/test_split'

    INPUT_MANIFEST = 'CHDIR_495_2022-05-07_19_whisper_zero.json'
    OUTPUT_MANIFEST = 'CHDIR_495_2022-05-07_19_word_level.json'

    e = ExtractSingleWord(
        input_manifest=os.path.join(ROOT, INPUT_MANIFEST),
        output_manifest=os.path.join(ROOT, OUTPUT_MANIFEST),
    )()