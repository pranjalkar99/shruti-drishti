import os
from shutil import copy2
from sklearn.model_selection import train_test_split


def videoprocessing(path, mp4path):
    for root, _, file in os.walk(path):
        for filename in file:
            if filename.endswith(".MOV"):
                print(root.split('/')[-1])
                print(root)
                os.makedirs(os.path.join(mp4path, root.split('/')[-1], "mp4"), exist_ok=True)
                base_filename, _ = os.path.splitext(filename)
                output_filename = f"{base_filename}.mp4"
                input_filepath = os.path.join(root, filename)
                # os.mkdir(os.path.join(root, "mp4")
                output_filepath = os.path.join(mp4path, root.split('/')[-1], output_filename)
                os.system(f"ffmpeg -i {input_filepath} {output_filepath}")
                print(f"Converted {filename} to {output_filename}")

def split_video_data(mp4folder, output_folder, train_size=0.8, val_size=0.1, test_size=0.1):
    all_videos = {}  # Dictionary to store videos per category
    for category in os.listdir(mp4folder):
        category_path = os.path.join(mp4folder, category)
        if os.path.isdir(category_path):
            category_videos = []
            for root, _, files in os.walk(category_path):
                for file in files:

                    video_path = os.path.join(root, file)
                    category_videos.append(video_path)
            all_videos[category] = category_videos

    # Print the number of videos found in each category for debugging
    print("Number of videos found per category:")
    for category, videos in all_videos.items():
        print(f"{category}: {len(videos)}")

    if not all_videos:
        raise ValueError("No video files found in the input folder.")

    # Split videos within each category
    for category, videos in all_videos.items():
        train_videos, test_val_videos = train_test_split(videos, train_size=train_size, random_state=42)
        val_videos, test_videos = train_test_split(test_val_videos, test_size=test_size / (test_size + val_size), random_state=42)

        os.makedirs(os.path.join(output_folder, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'val', category), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'test', category), exist_ok=True)

        for folder, video_list in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
            for video in video_list:
                output_path = os.path.join(output_folder, folder, category, os.path.basename(video))
                copy2(video, output_path)

if __name__ == "__main__":
    input_folder = "/home/bhaswata08/Documents/ISLV2/Data_init"  
    mp4path = "laura/mp4" # Dumb fucking logic, useless intermediate path for mp4s
    output_folder = "laura/output"
    videoprocessing(input_folder, mp4path)
    split_video_data(mp4path, output_folder)
    # TODO: add argparse