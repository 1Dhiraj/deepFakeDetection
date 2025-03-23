# Install dependencies if not already installed
# pip install opencv-python mtcnn tqdm

import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm

# Paths
base_dir = os.path.join(os.getcwd(), 'downloads')
real_videos_path = os.path.join(base_dir, 'original_sequences')
fake_videos_path = os.path.join(base_dir, 'manipulated_sequences')

print("Looking for real videos in:", real_videos_path)
print("Looking for fake videos in:", fake_videos_path)

output_faces_path = 'dataset/'
print("Real Videos Found:", os.listdir(real_videos_path))
print("Fake Videos Found:", os.listdir(fake_videos_path))

# Create output directories
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(output_faces_path, split, label), exist_ok=True)

# Initialize face detector
detector = MTCNN()

# Function to extract faces from video and save
def extract_faces_from_video(video_path, output_faces_path, label, split, max_faces=50):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_faces = 0

    while cap.isOpened() and saved_faces < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            # Handle negative box values safely
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]

            # Save face
            video_basename = os.path.basename(video_path).split('.')[0]
            face_filename = f"{video_basename}_{frame_count}.jpg"
            save_path = os.path.join(output_faces_path, split, label, face_filename)
            cv2.imwrite(save_path, face_crop)
            saved_faces += 1

            if saved_faces >= max_faces:
                break

        frame_count += 1

    cap.release()

# Process videos
def process_folder(folder_path, label):
    # Walk through subfolders and collect video files
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, f))

    print(f"Found {len(video_files)} {label} videos!")

    if len(video_files) == 0:
        print(f"No {label} videos found in {folder_path}!")
        return

    split_threshold = int(len(video_files) * 0.7)  # 70% train, 15% val, 15% test

    for idx, video_path in enumerate(tqdm(video_files)):
        if idx < split_threshold:
            split = 'train'
        elif idx < split_threshold + int(len(video_files) * 0.15):
            split = 'val'
        else:
            split = 'test'

        extract_faces_from_video(video_path, output_faces_path, label, split)

# Extract faces from real and fake videos
print("\nProcessing Real Videos...")
process_folder(real_videos_path, 'real')

print("\nProcessing Fake Videos...")
process_folder(fake_videos_path, 'fake')

print("\nFace Extraction Complete!")
