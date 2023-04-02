import sys
import os
import cv2
import random
from helpers.progress_bar import *

# Extract program arguments
video_folder_path = '../imulator_data/videos'
num_train_images = 210
num_val_images = 60
num_test_images = 30

# Set images path
images_path = '../yolo/train_data_test/images'
train_path = os.path.join(images_path, 'train')
val_path = os.path.join(images_path, 'val')
test_path = os.path.join(images_path, 'test')

# Create output folders
os.makedirs(images_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get numbert of videos
num_videos = len(os.listdir(video_folder_path))

# Calculate the number of images to extract from each video
num_train_images_per_video = num_train_images // num_videos
num_val_images_per_video = num_val_images // num_videos
num_test_images_per_video = num_test_images // num_videos

# Extract images from videos
l = len(os.listdir(video_folder_path))
printProgressBar(0, l, prefix = 'Extracting:', suffix = 'Complete', length = 50)
for idx, video_name in enumerate(os.listdir(video_folder_path)):
    # Skip if file is not .mp4
    if video_name.split('.', 1)[-1] != 'mp4':
        continue

    video_path = os.path.join(video_folder_path, video_name)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample random frames
    sample_frames = random.sample(
        range(0, num_frames),
        num_train_images_per_video + num_val_images_per_video + num_test_images_per_video
    )

    # Divide the sampled frames into train, val and test groups
    train_frames = sample_frames[0:num_train_images_per_video]
    val_frames = sample_frames[num_train_images_per_video:num_train_images_per_video + num_val_images_per_video]
    test_frames = sample_frames[num_train_images_per_video + num_val_images_per_video:]
    print(val_frames)
    # Extract the frames and save them into appropriate folders
    for i in range(0, len(train_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, train_frames[i])
        ret, frame = cap.read()
        train_img_path = os.path.join(train_path, video_name.partition('.')[0] + '_' + str(i) + '.jpg')
        cv2.imwrite(train_img_path, frame)

    sys.stdout.flush()

    for i in range(0, len(val_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val_frames[i])
        ret, frame = cap.read()
        val_img_path = os.path.join(val_path, video_name.partition('.')[0] + '_' + str(i) + '.jpg')
        cv2.imwrite(val_img_path, frame)

    sys.stdout.flush()

    for i in range(0, len(test_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frames[i])
        ret, frame = cap.read()
        test_img_path = os.path.join(test_path, video_name.partition('.')[0] + '_' + str(i) + '.jpg')
        cv2.imwrite(test_img_path, frame)

    sys.stdout.flush()

    cap.release()

    printProgressBar(idx + 1, l, prefix = 'Extracting:', suffix = 'Complete', length = 50)
