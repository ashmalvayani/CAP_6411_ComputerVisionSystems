# Python: Sample 10 videos per class and convert each to 480p MP4
import os, glob, subprocess

# Define the class names exactly as they appear in the dataset folders
classes = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone",
]

os.makedirs("data/HAR_subset", exist_ok=True)  # Directory for the subset

for cls in classes:
    src_dir = os.path.join("HAR-Dataset/HumanActivityRecognition-Dataset", cls)
    dst_dir = os.path.join("HAR-Dataset/HumanActivityRecognition-Prepared", cls)
    os.makedirs(dst_dir, exist_ok=True)
    # Get list of video files in the class directory (filter common extensions)
    videos = sorted(glob.glob(os.path.join(src_dir, "*.*")))
    # Take the first 10 videos (you can also randomize selection if desired)
    # for vid_path in videos[:10]:
    for vid_path in videos:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(dst_dir, base_name + ".mp4")
        # Convert to 832x480 MP4 (H.264 encoding)
        subprocess.run([
            "ffmpeg", "-y", "-i", vid_path,
            "-vf", "scale=832:480",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an", out_path
        ])
        print(f"Converted {vid_path} -> {out_path}")


for cls in classes:
    dst_dir = os.path.join("HAR-Dataset/HumanActivityRecognition-Prepared", cls)
    for vid_path in glob.glob(os.path.join(dst_dir, "*.mp4")):
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        caption_path = os.path.join(dst_dir, base_name + ".txt")
        
        # Use class name as the caption (e.g. "A person walking while reading a book.")
        caption = f"A person {cls.lower()}."
        
        with open(caption_path, "w") as f:
            f.write(caption)
        
        print(f"Captioned {vid_path} -> {caption}")