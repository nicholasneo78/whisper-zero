import os
from typing import List, Dict
from jiwer import cer, wer, mer
import json

import logging

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class CER:
    
    '''
    CER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return cer(reference=self.references, hypothesis=self.predictions)
    
class WER:
    
    '''
    WER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return wer(reference=self.references, hypothesis=self.predictions)
    
class MER:
    
    '''
    MER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return mer(reference=self.references, hypothesis=self.predictions)

class WERFromJSON:
    
    '''
    to get the WER from the JSON file with key "prediction" and "ground truth" after running the evaluate_model.py script to generate the json file
    '''

    def __init__(self, manifest_path:str) -> None:
    
        '''
        input_json_dir (str): the json directory that was generated from evaluate_model.py
        '''

        self.manifest_path = manifest_path

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
        

    def get_wer_result(self) -> None:

        '''
        main method to print the WER and CER
        '''

        # load the dict list that was loaded from the json manifest file
        data = self.load_manifest_nemo(input_manifest_path=self.manifest_path)

        # get the pred and the ground truth list
        pred_list = [pred['pred_str'] for pred in data]
        ground_truth_list = [ref['text'] for ref in data]

        # compute the WER
        get_wer = WER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_cer = CER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_mer = MER(
            predictions=pred_list,
            references=ground_truth_list
        )

        print()
        logging.getLogger('INFO').info("Test WER: {:.5f}".format(get_wer.compute()))
        logging.getLogger('INFO').info("Test CER: {:.5f}".format(get_cer.compute()))
        logging.getLogger('INFO').info("Test Word Acc: {:.5f}\n".format(1-get_mer.compute()))


    def __call__(self):
        return self.get_wer_result()
    
if __name__ == '__main__':

    ROOT = '/datasets/mms/transcribed/mms_transcribed_batch_2/test_split/'
    MANIFEST_PATH = 'test_manifest_495_with_pred.json'

    w = WERFromJSON(
        manifest_path=os.path.join(ROOT, MANIFEST_PATH)
    )()