import os
import subprocess
import argparse
from tqdm import tqdm
from dataset_loader import DatasetLoader

parser = argparse.ArgumentParser(description='Openpose CLI Wrapper')
parser.add_argument('--save_dir', default='/proj/brizk/output/openpose', help='Dir to save json results')
parser.add_argument('--verbose_dividend', default=10, type=int, help='openpose cli_verbose dividend of num of frames')
parser.add_argument('--ffmpeg_log_level', default='panic', type=str, help='ffmpeg log level')
parser.add_argument('--num_gpu', default=2, type=str, help='num of to be used gpus')
parser.add_argument('--num_gpu_start', default=0, type=str, help='index of first gpu to utilize')
# parser.add_argument('--fps', default=6, type=int, help='fps')

args = parser.parse_args()

p = subprocess.run(
    ['ffmpeg', '-hide_banner', '-loglevel', args.ffmpeg_log_level],
)
print(f'Setting ffmpeg log level to {args.ffmpeg_log_level}: {p}')    


if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
    
logger_file = open(f"{args.save_dir}/recent_run_log.txt", "w")


videos_loader = DatasetLoader()
for video_num, video in enumerate(tqdm(videos_loader, desc='Dataset (Videos)')):
    verbose_step = video.num_of_selected_frames//args.verbose_dividend
    print('-----------------------------------')
    print(f'Starting processing video: {video.filepath} verbose every {verbose_step}')
    logger_file.write(f'starting video {video_num}: {videos_loader.current_ds}/{video.name}\n')
    logger_file.flush()

    save_path = os.path.join(args.save_dir, video.name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        # print(f'Creating Directory {save_path}')
    else:
        num_of_frames_finished = len(os.listdir(save_path))
        if video.num_of_selected_frames == num_of_frames_finished:
            print(f'Skipping {save_path} as already exists and completed')
            logger_file.write(f'Skipping {save_path} as already exists\n')
            logger_file.flush()
            continue
        print(f'Directory {save_path} already exists, but will be overwritten')
        logger_file.write(f'Directory {save_path} already exists, but will be overwritten\n')
        logger_file.flush()


    p = subprocess.run(
        [
            './build/examples/openpose/openpose.bin',
            '--video', video.filepath,
            '--write_json', save_path,
            '--num_gpu', str(args.num_gpu),
            '--num_gpu_start', str(args.num_gpu_start),
            '--cli_verbose', str(verbose_step),
            '--frame_step', str(video.step),
            '--face', '--hand',
            '--display', '0',
            '--render_pose', '0',
        ],
        cwd='/proj/manoj/openpose'
    )
    # p.wait()
    print('{video.name} Subprocess:', p)

logger_file.close()
