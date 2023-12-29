import os, glob, cv2, argparse, yaml, json
from tqdm import tqdm
# REAL_DIR = ['/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/dfdc_ximi', 
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/inhouse-1_ximi', 
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/inhouse-2_ximi',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_real_1',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_real_2',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_real_3',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_real_4',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_real_5']

# FAKE_DIR = ['/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/dfdc_fakes_ximi',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_fakes_1',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_fakes_2',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_fakes_3',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_fakes_4',
#             '/home/surbhi/gautam/deep_fake_new/deepfake-detection/data/preprocessed/data/kartik_fakes_5']

# real_videos = []
# for e in REAL_DIR:
#     real_videos.extend(glob.glob(e + '/**/*.mp4'))

# print ("Real videos: ", len(real_videos))

# fake_videos = []
# for e in FAKE_DIR:
#     fake_videos.extend(glob.glob(e + '/**/*.mp4'))
# print ("Fake videos: ", len(fake_videos))


# from sklearn.model_selection import train_test_split

# train_real_videos, val_real_videos = train_test_split(real_videos, test_size=0.1)
# print ("Training real videos: ", len(train_real_videos))


# train_fake_videos, val_fake_videos = train_test_split(fake_videos, test_size=0.1)
# print ("Training fake videos: ", len(train_fake_videos))

# json_data = []
# for e in train_real_videos:
#     json_data.append({'video_path': e, 'label': 0, 'split': 'train'})
# for e in train_fake_videos:
#     json_data.append({'video_path': e, 'label': 1, 'split': 'train'})
# for e in val_real_videos:
#     json_data.append({'video_path': e, 'label': 0, 'split': 'val'})
# for e in val_fake_videos:
#     json_data.append({'video_path': e, 'label': 1, 'split': 'val'})

# import json
# with open('../metadata.json', 'w') as f:
#     json.dump(json_data, f)


def extract_frames(video_path):
    data = []
    reader = cv2.VideoCapture(video_path)
    # frame_count =0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
       
        data.append(frame)
    return data

def read_frames(video_element):
    label = video_element['label']
    frames = extract_frames(video_element['video_path'])
    # Calculate the interval to extract the frames
    frames_number = len(frames)
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)

    
    if video_element['split'] == 'val':
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames = frames[::frames_interval]
    return frames

def main():
    with open('../metadata.json', 'r') as f:
        paths = json.load(f)
    
    # sample = paths[0]
    for sample in tqdm(paths):
        frames = read_frames(sample)
        # print ('frames : ', len(frames))
        base_dir = os.path.dirname(sample['video_path'])
        fname = os.path.basename(sample['video_path'])
        frame_dir = os.path.join(base_dir, fname.split('.mp4')[0])

        # print ("path to save: ", frame_dir)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        for idx, frame in enumerate(frames):
            
            frame_path = os.path.join(frame_dir, f'frame_{idx}.png')
            # print ('Writing to :', )
            cv2.imwrite(frame_path, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
   
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    main()