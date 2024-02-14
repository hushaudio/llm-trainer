import whisper
from audio import download_audio
from moviepy.editor import AudioFileClip
from pytube import YouTube

import os

# Load the model
model = None
model_name = "large-v3"

def transcribe_audio(audio_file: str, _model_name:str = None):
    # make model available globally
    global model
    global model_name

    # Load the model
    if(_model_name): 
        model = whisper.load_model(_model_name)
        model_name = _model_name

    if model is None: 
        print("\033[94mNo model, loading\033[0m")
        model = whisper.load_model(model_name)
        print("\033[92mModel loaded!\033[0m")

    # Transcribe the audio
    result = model.transcribe(audio_file)
    return result['text']

def transcribe_youtube_videos(youtube_video_urls: list, _model_name:str = None, transcribe: bool = True, download: bool = True, caption_only: bool = False):
        
    folder = "audio"
    if(caption_only):
        return get_all_transcripts(youtube_video_urls, folder='transcripts')
    
    if download:
        get_all_audio(youtube_video_urls, folder=folder)
    

    if transcribe:
        transcribe_all_audio(model_name, folder=folder)

def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in minutes.
    """
    with AudioFileClip(file_path) as audio:
        return audio.duration / 60  # convert seconds to minutes

def split_audio(file_path: str, chunk_length: int, folder: str, filename: str) -> list:
    """
    Split an audio file into chunks of the specified length (in minutes).
    """
    chunk_files = []
    with AudioFileClip(file_path) as audio:
        total_duration = audio.duration  # duration in seconds
        for start in range(0, int(total_duration), chunk_length * 60):
            end = min(start + chunk_length * 60, total_duration)
            chunk_file = f"{folder}/{filename}_{start}_{end}.mp3"
            audio.subclip(start, end).write_audiofile(chunk_file, codec="mp3")
            chunk_files.append(chunk_file)
    return chunk_files

def get_all_audio(youtube_video_urls: list, folder: str = "audio"):

    # new list to store the results
    results = []
    count = 0
    # create a for loop to iterate through the list of youtube video urls
    for url in youtube_video_urls:
        try:
            count = count+1
            print("\033[95m-> Downloading audio from YouTube video " + str(count) + " of " + str(len(youtube_video_urls)) + "...\033[0m")
            # Download the audio from the YouTube video
            audio_file = download_audio(url, folder=folder, prefix=count)
            if audio_file is None:
                continue
            
            # if audio file is longer than 30 minutes append the results with the audio file
            duration = get_audio_duration(audio_file)
            if duration > 120:
                print("\033[95m-> Splitting audio into 120 minute chunks...\033[0m")
                chunk_files = split_audio(audio_file, 120, folder, audio_file.split("/")[-1].split(".")[0])
                results.extend(chunk_files)
            else:
                results.append(audio_file)
        except Exception as e:
            print("Error:", str(e))
            continue

    return results

def get_all_transcripts(youtube_video_urls: list, folder: str = "transcripts"):

    # new list to store the results
    results = []
    # create a for loop to iterate through the list of youtube video urls
    for url in youtube_video_urls:
        
        # Create a YouTube object
        yt = YouTube(url)

        title = yt.streams[0].title.replace(" ", "_")
        
        # Get English captions
        caption = yt.captions.get_by_language_code('en')

        # If captions are available, print them
        if caption:
            # save to file in folder, file is called {title}.txt
            filename = title + ".txt"
            filepath = folder + "/" + filename
            if os.path.exists(filepath):
                print(f"\033[91mTranscript {filename} already exists! Skipping...\033[0m")
                continue
            with open(filepath, "w") as f:
                f.write(caption.generate_srt_captions())
                print(f"\033[93mTranscript {filename} saved!\033[0m")
                
        else:
            print("No English captions found")

    return results


def transcribe_all_audio(_model_name: str, folder: str):
    # get all files in the audio folder
    files = os.listdir(folder)
    # filter mp3 only
    files = [folder + "/" + file for file in files if file.endswith(".mp3")]

    # new list to store the results
    results = []

    # make model available globally
    global model
    global model_name

    model_loaded = False

    # create a for loop to iterate through the list of youtube video urls
    for audio_file in files:

        ## get the filename from the path, it will be the very last item in the array when you split the path by "/"
        filename = audio_file.split("/")[-1].split(".")[0]
        ## now sanitize the filename
        filename =  filename + ".txt"
        filepath = "transcripts/" + filename
        # Transcribe the audio
        print(f"\033[93mtranscribing audio: {filepath}\033[0m")
        
        if os.path.exists(filepath):
            print(f"\033[91mTranscript {filename} already exists! Skipping...\033[0m")
            continue
        
        # handle model loading here 
        # to avoid loading the model 
        # when theres nothing to process
        if not model_loaded:    
            # Load the model
            if(_model_name): 
                model = whisper.load_model(_model_name)
                model_name = _model_name

            if model is None: 
                print("\033[94mNo model, loading\033[0m")
                model = whisper.load_model(model_name)
                print("\033[92mModel loaded!\033[0m")
                model_loaded = True

        try:
            result = model.transcribe(audio_file)
        except Exception as e:
            print("Error:", str(e))
            continue

        # Print the result
        print(result['text'])

        # create a txt file of the transcript
        with open(filepath, "w") as f:
            f.write(result['text'])
             
        # push the result to a list
        results.append(result['text'])
    
    # return the list
    return results

def unload_model():
    global model
    model = None