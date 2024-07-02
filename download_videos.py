import pandas as pd
from pytube import YouTube
import os
import json 

file_path = 'csv_files/Push-ups - From Front.csv'
data = pd.read_csv(file_path)

def download_video(link, output_path="videos", filename=None):
    yt = YouTube(link)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if filename:
        stream.download(output_path, filename=filename)
        return os.path.join(output_path, filename)
    else:
        stream.download(output_path)
        return os.path.join(output_path, stream.default_filename)


video_links = data['Link'].dropna().tolist()

video_paths = []
for idx, link in enumerate(video_links):
    try:
        row_name = f"{idx}_video.mp4"
        video_path = download_video(link, filename=row_name)
        video_paths.append(video_path)
    except Exception as e:
        print(f"Failed to download video {link}: {e}")

print("Downloaded videos with new filenames:", video_paths)
