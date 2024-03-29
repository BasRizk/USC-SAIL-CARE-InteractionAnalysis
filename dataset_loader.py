# -*- coding: future_fstrings -*- 
import os
import subprocess
import pickle
from tqdm import tqdm
from math import ceil
import cv2

class DatasetLoader:

    def __init__(self,
                 videos_paths_pkl="/mnt/care/BOSCCProject2/BOSCC_Project_Simons/simons_video_paths.pkl",
                 keys_to_exclude=['NYU'],
                 fps=6):
                 
        with open(videos_paths_pkl, 'rb') as f:
            filepath_dict = pickle.load(f)
            
        for k in keys_to_exclude:
            filepath_dict.pop(k)
            
        self.filepaths =\
            [(p, k) for k in filepath_dict for p in filepath_dict[k]]
            
        # TODO REPLACE FILEPATHS! REMOVE REDUNDANCY
        self.filepath_dict = {}
        for p, k in self.filepaths:
            filename = os.path.basename(p).split(".")[0]
            self.filepath_dict[(filename, k)] = p
        self.num_of_videos = len(self.filepaths)
        self._generator = self._videos_generator()
        self.fps = fps

    def __getitem__(self, item):
        ds, video_name = item
        filepath = self.filepath_dict[(video_name, ds)]
        self.current_ds = ds
        return VideoImgsGenerator(filepath, self.fps)
        
        
    def __len__(self):
        return self.num_of_videos

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)
    
    def _videos_generator(self):
        for self.current_file_num, filedata in enumerate(self.filepaths):
            filepath, self.current_ds = filedata
            try:
                yield VideoImgsGenerator(filepath, self.fps)
            except ZeroDivisionError:
                print(f'File {filepath} may be corrupted and is to be skipped')

    def close(self):
        cv2.destroyAllWindows()


class VideoImgsGenerator:

    def __init__(self, filepath, fps):
        self.filepath = filepath
        self.fps = fps
        self.name = filepath.split("/")[-1].split(".")[0]    
        self.video_cap = cv2.VideoCapture(filepath)
        self.org_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.org_frame_count =\
             int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps is not None:
            self.step = round(self.org_fps)//fps
        else:
            self.step = 1
        
        self.num_of_selected_frames = ceil(self.org_frame_count/self.step)
        self.length = self.org_frame_count/self.org_fps
        self.setup_img_generator()
        # print("total frame count", self.current_video_frame_count, 'length', length)
    
    def setup_img_generator(self):
        self.current_frame_num = -1
        self.current_frame_img = None
        self._generator = self._img_generator()
    
    def __iter__(self):
        return self

    def __len__(self):
        return self.num_of_selected_frames

    def __next__(self):
        # NOTE: frames are 0-indexed
        # Generating based on set fps
        while(True):
            self.current_frame_img = next(self._generator)
            self.current_frame_num += 1
            if (self.current_frame_num % self.step == 0) or\
                self.current_frame_img is None:
                break 
        return self.current_frame_img
        
    def __getitem__(self, frame_num):
        # NOTE: frames are 0-indexed
        if frame_num < 0:
            raise IndexError(f'Accepted only range between 0:{self.org_frame_count}')

        # Reset the generator if already passed the frame
        if self.current_frame_num > frame_num:
            # print('reseting generator')
            self.setup_img_generator()
            
        frame_diff = frame_num - self.current_frame_num
        disable_pbar = False if frame_diff > 500 else True
        
        if not disable_pbar:
            try:
                # print('trying to seek')
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self.current_frame_num = frame_num
                self.current_frame_img = next(self._generator)
                return self.current_frame_img
            except:
                # Loop over instead!
                pass
            
        # print('iterate')
        pbar = tqdm(
            total=frame_diff,
            disable=disable_pbar,
            desc=f'From frame {self.current_frame_num} to {frame_num}'
        )
        
        while self.current_frame_num < frame_num:
            self.current_frame_img = next(self._generator)
            self.current_frame_num += 1
            pbar.update(1)
                
        return self.current_frame_img
            
    def _img_generator(self):
        while(True):   
            ret, frame = self.video_cap.read()        
            if not ret:
                self.video_cap.release()
                break 
            yield frame


def write_video_into_img(filepath_dict):                
    save_directory = 'data'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    paths_file = open(f'{save_directory}/filepaths_list.txt', 'w')

    videos_filepaths = [(p, k) for k in filepath_dict for p in filepath_dict[k]]
    for video_filepath, dataset in videos_filepaths:
        saved_filepaths = ffmpeg_video_to_imgs(video_filepath, f'{save_directory}/{dataset}')
        paths_file.write("\n".join(saved_filepaths))
        paths_file.write("\n")
        paths_file.flush()
    paths_file.close()

def ffmpeg_video_to_imgs(video_filepath, save_path_prefix, fps=3):
    video_name = video_filepath.split("/")[-1].split(".")[0]
    save_path_prefix += f'/{video_name}'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    saved_filepath_format = f'{save_path_prefix}/{video_name}_%d.jpg'  

    length = int(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", video_filepath],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT).stdout))
    num_of_imgs = length*2
    subprocess.call(['ffmpeg', '-i', video_filepath, '-vf', f'fps={fps}', saved_filepath_format])
    print(f'Saved to {save_path_prefix} {num_of_imgs} frames')
    saved_filepaths = [saved_filepath_format % i for i in range(num_of_imgs)]
    return saved_filepaths

