# Zephyr_YT_Trainer_mixed.ipynb README

## Overview
`Zephyr_YT_Trainer_mixed.ipynb` is a Jupyter notebook that forms part of the YouTube Trainer Application, facilitating the process of creating a training dataset from YouTube videos for language models. This notebook extends the functionality by allowing the incorporation of external datasets to the generated data.

## Prerequisites
- Jupyter or a compatible environment to run `.ipynb` files.
- audio.py, trascriner.py, and data_creator.py files from the YouTube Trainer Application repository.
- Python 3.11 environment with required dependencies installed (see `requirements.txt`).
- An active internet connection for downloading data and models.

## Setup
Before you begin, ensure to follow all the setup steps in the main repository's `README.md` to prepare your environment and install the necessary dependencies.

## Features
- Combines YouTube audio transcripts with external datasets.
- Processes text data for training language models.
- Configurable data chunking for model training optimization.
- Training session monitoring with WandB.
- Any file located in the transcription folder can be used to create datasets.

## Usage Instructions
1. **Initial Configuration**: At the top of the notebook, set your configuration variables. This includes paths, model names, and data chunk sizes.
2. **Data Downloading and Processing**: Follow the cells to download YouTube videos, extract audio, and transcribe using Whisper models.
3. **Data Enhancement**: Optionally, add external datasets from Huggingface to your data to increase diversity or introduce new capabilities such as function calling.
4. **Model Training**: Train your model with the prepared data. Monitor the training process using WandB to keep track of the performance.
5. **Evaluation and Testing**: After training, use the provided cells to evaluate the performance of your model.

## Customization
- `model_name`: Choose from various Whisper models for transcription based on your accuracy and performance needs.
- Data chunking parameters can be adjusted to optimize the training data for your specific model's requirements.
- Integration of external datasets can be done by modifying the relevant sections in the notebook.

## Post-Training
The notebook includes cells to save the trained model. Follow the instructions to test the model and then save it for future use or deployment.

## Support
If you encounter any issues or have questions about the notebook, please refer to the main repository's `README.md` for guidance and troubleshooting tips.
