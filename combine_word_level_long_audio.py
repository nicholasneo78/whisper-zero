"""
combine the words into utterances level based on the ground truth
"""

import os
import json
from tqdm import tqdm
from typing import Dict, List

class CombineWordToUtterances:

    '''
    main class to do the combination
    '''

    def __init__(self, ref_manifest: str, word_level_manifest: str, output_manifest: str, language: str) -> None:
        self.ref_manifest = ref_manifest
        self.word_level_manifest = word_level_manifest
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
    
    def combine_word_level_to_utt(self) -> None:

        '''
        main method to combine the words into utterances based on the reference manifest
        '''

        # load both the manifests
        ref_manifest = self.load_manifest_nemo(input_manifest_path=self.ref_manifest)
        word_level_manifest = self.load_manifest_nemo(input_manifest_path=self.word_level_manifest)

        word_level_idx = 0
        
        for entry_ref in tqdm(ref_manifest):
            utterance_word_list = []

            # check if the start of the word level entry falls between the range in the ref entry
            while entry_ref['start'] > word_level_manifest[word_level_idx]['start']:
                word_level_idx += 1

            while entry_ref['start'] <= word_level_manifest[word_level_idx]['start']:

                utterance_word_list.append(word_level_manifest[word_level_idx]['text'].lstrip())
                word_level_idx += 1

                if entry_ref['end'] < word_level_manifest[word_level_idx]['end']:
                    break

            