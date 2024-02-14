# YouTube LLM Trainer Application README

## Overview
The YouTube LLM Trainer Application automates the creation of training datasets for language models. It downloads YouTube videos, extracts audio, transcribes it using OpenAI's Whisper models, and processes the transcripts into structured training data for machine learning models, with a focus on function calling.

## Prerequisites
- Tested on Python 3.11
- `ffmpeg` for audio processing
- An internet connection to download necessary models and data
- WSL is recommended for Windows users

## Installation

1. Create a conda environment with Python 3.11 and activate it.

```bash
conda create -n youtube-trainer python=3.11
conda activate youtube-trainer
```

2. Clone the repository and install the required Python packages, ensuring that `ffmpeg` is installed on your system for audio conversion tasks.

3. Clone llama.cpp to to the the root folder of this repo and run the following code in the conda environment

 ```bash
 git clone https://github.com/ggerganov/llama.cpp.git
 pip install -r ./llama.cpp/requirements.txt` 
 ```
 
4. Install the required Python packages.

```bash
pip install -r requirements.txt
```

## Repository Structure
- `audio.py`: Handles YouTube video downloads and audio conversions.
- `transcriber.py`: Transcribes audio using Whisper models, which are downloaded by the script.
- `data_creator.py`: Segments transcripts and prepares training data.
- `Zephyr_YT_Trainer_mixed.ipynb`: Enhances training data by incorporating external datasets.
- `Zephyr_FC_Trainer.ipynb`: Trains a function caller model.
- `Zephyr_lora_to_model.ipynb`: Merges LoRA with the base model and configures it for efficiency.
- `README.md`: Provides documentation and usage instructions for the application.

## Whisper Model Selection
Choose a transcription model from the following options available within the application:
- `large-v3`
- `large`
- `small.en`
- `base`
- `tiny.en`
- `medium.en`
- `medium`
- `tiny`
- `small`
- `base.en`

Select the desired model in `transcriber.py` by updating the `model_name` variable accordingly.

## Workflow

### Audio Processing
Utilize `audio.py` for downloading videos and converting them to audio.

### Transcription
Use `transcriber.py` to transcribe audio files with the selected Whisper model.

### Data Preparation
Execute `data_creator.py` to segment transcripts into chunks and form a training dataset.

### Training Data Generation
Generate a dataset with `{question: string, answer: string}` formatting in the `Zephyr_YT_Trainer_mixed.ipynb` or `Zephyr_FC_Trainer.ipynb` notebooks.

### Model Enhancement and Training
For merging LoRA with the base model and applying bits and bytes optimization, utilize `Zephyr_lora_to_model.ipynb`.

### Post-Training
Test the trained model's performance and save it for future use.

## Advanced Configuration
Check and adjust the configuration variables at the top of each notebook to suit your project requirements.

## Model Training and Function Calling
`Zephyr_FC_Trainer.ipynb` is specifically designed for training models capable of function calling using the `glaive-function-calling-v2-zephyr` dataset.

## Testing and Validation
Post-training, assess the model's performance and create synthetic datasets for further validation.

## Final Notes
Review the settings and system prompts in the `data_creator.py` to align with your objectives. This script also includes the function definition for the API call.

---

### Disclaimer
This application has been developed for research purposes only and is meant for academic and educational use in machine learning and natural language processing.

### Restrictions on Use
It is the user's responsibility to adhere to YouTube's Terms of Service and the usage policies of any content they process with this tool. The application is not intended for commercial use and should only be used with videos that are permitted for such handling.
