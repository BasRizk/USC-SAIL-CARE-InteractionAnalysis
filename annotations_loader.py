# -*- coding: future_fstrings -*- 

import os
import glob
import pickle
import numpy as np
import pandas as pd

class AnnotationsLoader:
         
    # def __init__(self, faces_dir='/proj/brizk/output/retinaface',
    #              dataset_loader=None, files_location='*/*'):
    #     if dataset_loader is None:
    #         dataset_loader = DatasetLoader()
    #     self.dataset = dataset_loader.filepath_dict.keys()
        
    #     self.faces_dir = faces_dir
    #     self.num_of_files = len(dataset_loader.num_of_videos)
    #     self._faces_generator = self._faces_files_generator()
    
    
    def __init__(self, ds, video_name,
                 faces_dir='/proj/brizk/output/retinaface',
                 attention_dir='/proj/brizk/output/attentiontarget'):
        self.ds = ds
        self.video_name = video_name
        
        if faces_dir:
            self.faces_dir = faces_dir

            column_names = [
                'frame', 'confidence', 'left', 'top', 'right', 'bottom'
            ]
            self.faces_file = pd.read_csv(
                os.path.join(
                    *[self.faces_dir, self.ds, f'{self.video_name}.csv']
                ),
                header=None, names=column_names, usecols=range(6)
            )            
            self._faces_generator = self._faces_gen()
        
        if attention_dir:
            self.attention_dir = attention_dir
            with open(
                f'{self.attention_dir}/{self.ds}/{self.video_name}.pkl', 'rb'
            ) as f:    
                self.attention_file = np.array(pickle.load(f))
            self._attention_generator = self._attention_gen()
        
    def __iter__(self):
        return self

    def __next__(self):
        
        
        faces_per_frame = next(self._faces_generator)
        attention_per_frame = next(self._attention_generator)
        indices = faces_per_frame.index
        return {
            'frame': faces_per_frame.loc[indices[0], 'frame'],
            'faces': faces_per_frame,
            'attention': attention_per_frame,
        }
    
    def __getitem__(self, frame_num):
        selection = self.faces_file['frame'] == frame_num
        # NOTE: Assuming that faces and attention lists
        # are of the matching in length
        faces_per_frame = self.faces_file[selection]
        attention_per_frame = self.attention_file[selection]
        indices = faces_per_frame.index
        return {
            'frame': faces_per_frame.loc[indices[0], 'frame'],
            'faces': faces_per_frame,
            'attention': attention_per_frame
        }
        
        
    def _faces_gen(self):
        frame_collection = []
        for i in self.faces_file.index:
            if not frame_collection:
                frame_collection = [i]  
            elif frame_collection[-1] == self.faces_file.loc[i, 'frame']:
                frame_collection.append(i)
            else:
                yield self.faces_file.loc[
                    frame_collection[0]:frame_collection[-1]
                ]
                frame_collection = [i]

        if frame_collection:
            yield self.faces_file.loc[
                frame_collection[0]:frame_collection[-1]
            ]


    def _attention_gen(self):
        frame_collection = []
        for inference in self.attention_file:
            if not frame_collection:
                frame_collection = [inference]
            elif frame_collection[-1] == inference['frame']:
                frame_collection.append(inference)
            else:
                yield frame_collection
                frame_collection = [inference]
        
        if frame_collection:
            yield frame_collection
        

class RetinafaceInferenceGenerator:
    
    def __init__(self, faces_dir='/proj/brizk/output/retinaface', files_location='*/*'):
        self.faces_dir = faces_dir
        self.num_of_files = len(glob.glob(f'{faces_dir}/{files_location}'))
        self._generator = self._faces_files_generator()
    
    def __iter__(self):
        return self

    def __len__(self):
        return self.num_of_files

    def __next__(self):
        return next(self._generator)
            
    def _faces_files_generator(self):
        for ds in os.listdir(self.faces_dir):
            for filename in os.listdir(os.path.join(*[self.faces_dir, ds])): 
                yield (ds, filename)
                