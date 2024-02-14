from pytube import YouTube
import subprocess
import os
from data_creator import sanitize_filename 

def convert_mp4_to_mp3(input_file, output_file):
    # check if output_file exists by checking the file path
    # if it does, return the path to the file
    if os.path.exists(output_file):
        return output_file
    
    command = ['ffmpeg', '-i', input_file, '-vn', '-ab', '128k', '-ar', '44100', '-y', '-loglevel', 'panic', output_file]
    subprocess.run(command)
    return output_file

# Function to download audio from YouTube video
def download_audio(url, folder="audio", prefix=""):
    try:
        print("Getting Video from YouTube to extract audio...")
        # Create a YouTube object with the video URL
        video = YouTube(url)
        
        title = video.streams[0].title.replace(" ", "_")
        
        print("\033[93mChecking if audio file already exists...\033[0m")
        print(f"Title: {title}")

        ## rewrite this to replace all | in the string filepath = f"./{folder}/{title.replace('|', '')}.mp3" also remove any trailing dots.  Also completely sanitize it as a filepath would be when saving with os
        sanitized = sanitize_filename(title)
        filepath = f"./{folder}/{prefix}{sanitized}.mp3"

        print(f"filepath: {filepath}")

        if os.path.exists(filepath):
            print(f"\033[91mAudio file {filepath} already exists! Returning path to mp3 file...\033[0m")
            return filepath
        
        # print in green color (92m) download does not exist downloading
        print("\033[92mAudio file does not exist, downloading...\033[0m")

        # Get the audio stream of the video
        audio_stream = video.streams.filter(only_audio=True).first()

        # Download the audio stream
        result = audio_stream.download("downloads/temp")

        print("\033[92mDownload complete!\033[0m")

        ## result is the path to the downloaded audio file
        ## get the filename from the path, it will be the very last item in the array when you split the path by "/"
        filename = result.split("/")[-1].split(".")[0]

        ## now sanitize the filename
        filename = sanitize_filename(filename)

        print("\033[94mConverting mp4 to mp3...\033[0m")

        path = convert_mp4_to_mp3(result, f"./{folder}/{filename}.mp3")

        print("\033[92mConverted to mp3! Deleting video file...\033[0m")
        # now delete the original mp4 file (result)

        subprocess.run(['rm', result])
        print("\033[93mDeleted video file! Returning path to mp3 file...\033[0m")
        return path

    except Exception as e:
        print("\033[94mError: " + str(e) + "\033[0m")
        return False