import os
import json
from dotenv import load_dotenv
from transformers import GPT2Tokenizer
import re
import aiofiles

import nest_asyncio
import asyncio
from asyncio import Semaphore

nest_asyncio.apply()

asyncio.get_event_loop().set_debug(True)

from typing import TypedDict, List, Any, Dict

# Load the .env file
load_dotenv()

# OpenAI's client setup
from openai import AsyncOpenAI
client = AsyncOpenAI()

# Define the size of each chunk and the overlap for creating training data
CHUNK_SIZE = 2000
OVERLAP_SIZE = 150

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def append_to_json_file(data_file_path: str, new_data: Dict[str, Any]) -> None:
    # Read the existing data
    try:
        with open(data_file_path, 'r') as file:
            existing_data: List[Dict[str, Any]] = json.load(file)
    except FileNotFoundError:
        existing_data = []

    # new_data should be an array of objects with 'question' and 'answer' keys
    if not isinstance(new_data, list):
        raise ValueError('new_data should be an array of objects with \'question\' and \'answer\' keys')
    
    # Validate the data
    for item in new_data:
        if not isinstance(item, dict):
            return print('new_data should be an array of objects with \'question\' and \'answer\' keys')
        elif 'question' not in item or 'answer' not in item:
            return print('new_data should be an array of objects with \'question\' and \'answer\' keys')
        elif not isinstance(item['question'], str) or not isinstance(item['answer'], str):
            return print('Invalid data type')

    # Append new data
    existing_data.append(new_data)

    # Write back to the file
    with open(data_file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


class SampleData(TypedDict):
    question: str
    answer: str

def is_sample_data(item) -> bool:
    return isinstance(item, dict) and 'question' in item and 'answer' in item

def create_dataset_json(folder_path: str) -> List[SampleData]:
    all_json: List[SampleData] = []
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                for item in json_data:
                    # Skip items with 'training_data' key and empty array
                    if isinstance(item, dict) and item.get('training_data') == []:
                        continue
                    elif isinstance(item, list) and all(is_sample_data(sub_item) for sub_item in item):
                        all_json.extend(item)
                    elif is_sample_data(item):
                        all_json.append(item)
                    else:
                        print(f'Unexpected data format in item: {item}')
                        continue

        for data in all_json:
            if not isinstance(data['question'], str) or not isinstance(data['answer'], str):
                print('Invalid data:', data)
                raise ValueError('Invalid data type')

        with open('./dataset.json', 'w', encoding='utf-8') as f:
            json.dump(all_json, f)

    except Exception as error:
        print('Error reading JSON files:', error)
    return all_json

# Function to chunk the transcript into specified sizes
def chunk_transcript(transcript, chunk_size=1000, overlap_size=100):
    # Tokenize the entire transcript
    tokens = tokenizer.tokenize(transcript)
    chunks = []

    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + chunk_size, len(tokens))
        chunk_tokens = tokens[start_index:end_index]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move the start index, ensuring overlap with the previous chunk
        start_index = end_index - overlap_size if end_index - overlap_size > start_index else end_index
    print("Total chunks: "+ str(len(chunks)))
    return chunks

def sanitize_filename(filename):
    """
    Sanitizes the input string to be used as a safe filename.
    Removes or replaces characters that are not suitable for filenames.
    Args:
        filename (str): The string to sanitize.
    Returns:
        str: A sanitized string suitable for use as a filename.
    """
    # Remove invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Optionally, you can also convert to lowercase
    sanitized = sanitized.lower()

    return sanitized

def save_to_json(data, file_path):
    """
    Saves the given data to a JSON file.
    Args:
        data (dict or list): The data to save.
        file_path (str): The path to the file where the data should be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

# Main function to process all transcripts in a folder
def fix_poisoned_training_data(chunk, filename, topic, retry = 0, retryLimit = 5, notes = ""):
    retry += 1
    print(f"Creating training data for chunk: {chunk[:100]}...")
    
    try:
        page_name = filename.replace(".txt", " ").replace("_", "")
        print("NOTERS: "+notes)
        system = {"role": "system", "content": f"""You are an expert in {topic} and any related topics. 
You specialize in creating datasets of question/answer and instruction/task completion, and prompt / response pairs that provide the helpful assistance required when seeking information from the provided transcripts. 
Your expertise allows you to contextualize each pair of training data, focusing on the nuances of the topic and related areas.  
We are working with data that got poisoned with bad phrasing or data, so we need to rewrite it.  For the most part, everything is correct, the main issue is as noted here by the user: {notes}
You will fix the data and use your internal information that you are more than 90% sure on to enhance the data and make it more accurate and robust for training a large language model on {topic}. 
We are creating the most effective large language model training data possible, so its important that these prompt/response pairs sometimes contain a list of instruction and completion pairs as well.  
You will mimic the initial intent of the training data, while removing the poisoned phrasing from it specified in the notes here: {notes}."""
        }
        prompt = {"role": "user", "content": f"""Please call the function "create_data" to expertly rebuild the highest quality large language model training data on the topic of {topic} for the following content.
Ignore content not relevant to the main topic or any related topics.
Create an exhaustive list of prompt/response pairs, ensuring they are relevant and informative.
The questions should be posed from the viewpoint of someone unfamiliar with the topic, and answers should be detailed and easily understandable.
Where ever possible, format the question and answer as instruction/completion pairs while still keeping the json in a questions and answers json structure.
When the response contains instructions, always use a numbered list in the response to give the user instructions on how to complete the task. 
Include any relevant information not mentioned in the transcript.
For instance if we are referring to an option in a window, make sure to include the name of the window or object we are setting properties on.
```transcript\n{chunk}\n```\n{notes}"""
        }
        
        # get tokens of chunk using the transformers library 
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", #"gpt-3.5-turbo-16k", # Replace with your specific model
            max_tokens = 4096 - 1350,
            
            messages=[
                system,
                prompt
            ],
            tool_choice="auto",
            tools=[{
                "type": "function",
                "function": {
                    "name": "create_training_data",
                    "description": f"Generates detailed question-answer pairs from the given content, covering all of the points made in the content given by the user about {topic} without missing any facts or information mentioned in the users prompt.  Only return this defined object structure.",
                    "required": ["training_data"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "training_data": {
                                "type": "array",
                                "description": f"An array of question-answer pairs where each pair thoroughly covers the topics in the text provided. They must relate to {topic}.  If there is no information relevant to {topic} or its related topics than you can return an empty array here.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": f"A detailed question that comprehensively covers up to two topics that pertain to {topic} in the provided text, leaving no aspect of the text that pertain to the {topic} topic unaddressed.  Do not mention 'the creator' or 'the author' or any reference do this being a transcription.  Only use relevant {topic} questions that work without any context"
                                        },
                                        "answer": {
                                            "type": "string",
                                            "description": f"A comprehensive answer that provides complete and accurate information in response to the question about {topic}. Do not mention 'the creator' or 'the author' or any reference do this being a transcription.  Only use relavant {topic} questions that work without any context"
                                        }
                                    }
                                }
                            },
                        },
                    }
                }
            }]
        )
        

        print(f"\033[94mReceived Response...\033[0m")
        return handle_response(response, chunk, filename, topic, retry, retryLimit, notes, prompt, system)
        
    except Exception as e:
        if retry < retryLimit:
            print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
            return create_training_data(chunk, filename, topic, retry)
        else:
            print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
            return []


def handle_response(response, chunk, filename, topic, retry, retryLimit):
    
        # Handle the response correctly
        if response.choices and len(response.choices) > 0:
            print(f"\033[92mHas choices...\033[0m")
            arguments_dict = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

            print(f"\033[92mHas arguments...\033[0m")
            # Extract the array from the dictionary
            training_data_array = arguments_dict.get("training_data", [])
            print(f"\033[92mParsed arguments...{training_data_array}\033[0m")
            
            # check if training data is in question/answer format - if not handle retry - in some cases it will be an object and not an array
            if not isinstance(training_data_array, list):
                print(f"\033[91mError in generating training data -> retrying\nError Message: {training_data_array}\033[0m")
                if retry < retryLimit:
                    return create_training_data(chunk, filename, topic, retry)

            # Save the training data to a JSON file 
            # Q: is this json? A: 
            saveResponse = {
                "system": system["content"],
                "prompt": prompt["content"], 
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            
            save_training_response(f"{sanitize_filename(topic)}_training_data.json", saveResponse)

            # Now you can use training_data_array as a regular Python list
            print(f"Training data array: {training_data_array}")
            return training_data_array
        else:
            if retry < retryLimit:
                print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
                return create_training_data(chunk, filename, topic, retry)
            else:
                print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
                return []
            
# Main function to process all transcripts in a folder
def create_training_data(chunk, filename, topic, retry = 0, retryLimit = 5, notes = ""):
    retry += 1
    print(f"Creating training data for chunk: {chunk[:100]}...")
    
    try:
        page_name = filename.replace(".txt", " ").replace("_", "")
        system = create_data_creator_system_prompt(topic)
        prompt = create_data_creator_prompt(page_name, chunk, notes)
        
        # get tokens of chunk using the transformers library 
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", #"gpt-3.5-turbo-16k", # Replace with your specific model
            max_tokens = 4096 - 1350,
            
            messages=[
                system,
                prompt
            ],
            tool_choice="auto",
            tools=[create_training_data_function_definition(topic)]
        )
        

        print(f"\033[94mReceived Response...\033[0m")
        
        # Handle the response correctly
        if response.choices and len(response.choices) > 0:
            print(f"\033[92mHas choices...\033[0m")
            arguments_dict = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

            print(f"\033[92mHas arguments...\033[0m")
            # Extract the array from the dictionary
            training_data_array = arguments_dict.get("training_data", [])
            print(f"\033[92mParsed arguments...{training_data_array}\033[0m")
            
            # check if training data is in question/answer format - if not handle retry - in some cases it will be an object and not an array
            if not isinstance(training_data_array, list):
                print(f"\033[91mError in generating training data -> retrying\nError Message: {training_data_array}\033[0m")
                if retry < retryLimit:
                    return create_training_data(chunk, filename, topic, retry)

            # Save the training data to a JSON file 
            # Q: is this json? A: 
            saveResponse = {
                "system": system["content"],
                "prompt": prompt["content"], 
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            
            save_training_response(f"{sanitize_filename(topic)}_training_data.json", saveResponse)

            # Now you can use training_data_array as a regular Python list
            print(f"Training data array: {training_data_array}")
            return training_data_array
        else:
            if retry < retryLimit:
                print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
                return create_training_data(chunk, filename, topic, retry)
            else:
                print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
                return []
        
    except Exception as e:
        if retry < retryLimit:
            print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
            return create_training_data(chunk, filename, topic, retry)
        else:
            print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
            return []

# Main function to process all transcripts in a folder
async def create_training_data_async(chunk, filename, topic, retry = 0, retryLimit = 5, notes = ""):
    retry += 1
    print(f"Creating training data for chunk: {chunk[:100]}...")
    try:
        page_name = filename.replace(".txt", " ").replace("_", "")
        system = create_data_creator_system_prompt(topic)
        prompt = create_data_creator_prompt(page_name, chunk, notes)
        
        # get tokens of chunk using the transformers library 
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview", #"gpt-3.5-turbo-16k", # Replace with your specific model
            max_tokens = 4096 - 1350,
            
            messages=[
                system,
                prompt
            ],
            tool_choice="auto",
            tools=[create_training_data_function_definition(topic)]
        )
        
        print(f"\033[94mReceived Response...\033[0m")
        
        # Handle the response correctly
        if response.choices and len(response.choices) > 0:
            print(f"\033[92mHas choices...\033[0m")
            arguments_dict = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

            print(f"\033[92mHas arguments...\033[0m")
            # Extract the array from the dictionary
            training_data_array = arguments_dict.get("training_data", [])
            print(f"\033[92mParsed arguments...{training_data_array}\033[0m")
            
            # check if training data is in question/answer format - if not handle retry - in some cases it will be an object and not an array
            if not isinstance(training_data_array, list):
                print(f"\033[91mError in generating training data -> retrying\nError Message: {training_data_array}\033[0m")
                if retry < retryLimit:
                    return create_training_data(chunk, filename, topic, retry)

            # Save the training data to a JSON file 
            # Q: is this json? A: 
            saveResponse = {
                "system": system["content"],
                "prompt": prompt["content"], 
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            
            save_training_response(f"{sanitize_filename(topic)}_training_data.json", saveResponse)

            # Now you can use training_data_array as a regular Python list
            print(f"Training data array: {training_data_array}")
            return training_data_array
        else:
            if retry < retryLimit:
                print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
                return create_training_data(chunk, filename, topic, retry)
            else:
                print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
                return []
        
    except Exception as e:
        if retry < retryLimit:
            print(f"\033[91mError in create_training_data -> retrying\nError Message: {e}\033[0m")
            return create_training_data(chunk, filename, topic, retry)
        else:
            print(f"\033[91mError in generating training data -> returning empty array\nError Message: {e}\033[0m")
            return []

# Main function to process all transcripts in a folder
def process_transcripts(folder_path: str, topic: str, notes: str, skip_existing=True):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    transcript = file.read()
                    chunk_and_process_data(transcript, filename, topic, skip_existing, notes=notes)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")   
    return sanitize_filename(topic)+".json"

async def chunk_and_process_data_async(transcript, filename, topic, skip_existing, notes: str, offset_first=False):
    chunks = chunk_transcript(transcript, CHUNK_SIZE, OVERLAP_SIZE)

    # File to store aggregated training data
    data_filename = f'{filename}_data.json'
    data_file_path = os.path.join('./data', data_filename)
    print(f"data_file_path: {data_file_path}")
    # if filepath exists, continue
    if skip_existing and os.path.exists(data_file_path):
        # print in red that we are skipping this file and include the filename
        print(f"\033[91mSkipping file {filename} because {data_filename} already exists...\033[0m")
        return None
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)

    ## remove first offset_first chunks from chunk list
    if offset_first is not False and len(chunks) > offset_first:
        chunks = chunks[offset_first:]

    # Prepare a list of tasks for each chunk without immediately awaiting them
    tasks = [create_training_data_async(chunk, filename, topic, notes=notes) for chunk in chunks]

    # Await all tasks concurrently. This ensures that all API calls are made at once, or as allowed by the event loop.
    training_data_results = await asyncio.gather(*tasks)

    # Process each training data result
    for training_data in training_data_results:
        if training_data and len(training_data) > 0:
            append_to_json_file(data_file_path, training_data)


async def handle_file(file_path, filename, topic, skip_existing, notes):
    async with aiofiles.open(file_path, 'r') as file:
        transcript = await file.read()
        # Process the transcript asynchronously
        await chunk_and_process_data_async(transcript, filename, topic, skip_existing, notes=notes)
            
# Define the function with an additional parameter for concurrency control
async def process_transcripts_async(folder_path: str, topic: str, notes: str, skip_existing=True, max_concurrent_files=4):
    # Initialize the semaphore with your desired concurrency limit
    sem = Semaphore(max_concurrent_files)
    tasks = []
    print(f"Processing files in {folder_path}...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            task = asyncio.create_task(handle_file_with_semaphore(sem, file_path, filename, topic, skip_existing, notes))
            tasks.append(task)
    await asyncio.gather(*tasks)

# Ensure the handle_file function is adapted to work with the semaphore
async def handle_file_with_semaphore(sem: Semaphore, file_path, filename, topic, skip_existing, notes):
    async with sem:
        await handle_file(file_path, filename, topic, skip_existing, notes) 

########################





def chunk_and_process_data(transcript, filename, topic, skip_existing, notes: str, offset_first = False):
    chunks = chunk_transcript(transcript, CHUNK_SIZE, OVERLAP_SIZE)

    # File to store aggregated training data
    data_filename = f'{filename}_data.json'
    data_file_path = os.path.join('./data', data_filename)
    print(f"data_file_path: {data_file_path}")
    # if filepath exists, continue
    if skip_existing and os.path.exists(data_file_path):
        # print in red that we are skipping this file and include the filename
        print(f"\033[91mSkipping file {filename} because {data_filename} already exists...\033[0m")
        return None
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)


    ## remove first offset_first chunks from chunk list
    if offset_first is not False and len(chunks) > offset_first:
        # log that you're offsetting
        print(f"\033[93mOffsetting {offset_first} chunks from file {filename}\033[0m")
        chunks = chunks[offset_first:]
    
    for i in range(len(chunks)):

        # Log in yellow which number chunk is being processed of how many chunks for the filename
        print(f"\033[93mProcessing chunk {i+1} of {len(chunks)} for file {filename}\033[0m")

        # Call the create_training_data function for each chunk
        training_data = create_training_data(chunk=chunks[i], filename=filename, topic=topic, notes=notes)
        if training_data and len(training_data) > 0:
            # log appending and filepath
            print(f"\033[92mAppending training data to {data_filename}...\033[0m")
            # Append the new data to the JSON file
            append_to_json_file(data_file_path, training_data)


def append_to_json_file(data_file_path: str, new_data: List[Dict[str, Any]]) -> None:
    try:
        # if filepath does note exist, create it and write empty array
        if not os.path.exists(data_file_path):
            with open(data_file_path, 'w') as file:
                json.dump([], file)
                
        # Read the existing data
        with open(data_file_path, 'r') as file:
            existing_data: List[Dict[str, Any]] = json.load(file)
    except FileNotFoundError:
        existing_data = []
        
    # Validate new data structure
    for item in new_data:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            print(f"Failed item for question and answer: \n{item}")
            raise ValueError(f"Each item in new_data must be a dict with 'question' and 'answer' keys: \n{new_data}")

    # Append new data to the existing data
    existing_data.extend(new_data)

    # Write back to the file
    with open(data_file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


def serialize_data(data):
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_data(v) for v in data]
    elif isinstance(data, (str, int, float, bool)):
        return data
    else:
        return str(data)  # Convert non-serializable types to string or a suitable format


def save_training_response(data_file_path: str, new_data):
    try:
        with open(data_file_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []

    # Serialize new_data before appending
    serializable_new_data = serialize_data(new_data)

    # Append new data to the existing data
    existing_data.append(serializable_new_data)

    # Write back to the file
    with open(data_file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

def create_training_data_function_definition(topic:str):
    return {
    "type": "function",
    "function": {
        "name": "create_training_data",
        "description": f"Generates detailed question-answer pairs from the given content, covering all of the points made in the content given by the user about {topic} without missing any facts or information mentioned in the users prompt.  Only return this defined object structure.",
        "required": ["training_data"],
        "parameters": {
            "type": "object",
            "properties": {
                "training_data": {
                    "type": "array",
                    "description": f"An array of question-answer pairs where each pair thoroughly covers the topics in the text provided. They must relate to {topic}.  If there is no information relevant to {topic} or its related topics than you can return an empty array here.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": f"A detailed question that comprehensively covers up to two topics that pertain to {topic} in the provided text, leaving no aspect of the text that pertain to the {topic} topic unaddressed.  Do not mention 'the creator' or 'the author' or any reference do this being a transcription.  Only use relevant {topic} questions that work without any context"
                            },
                            "answer": {
                                "type": "string",
                                "description": f"A comprehensive answer that provides complete and accurate information in response to the question about {topic}. Do not mention 'the creator' or 'the author' or any reference do this being a transcription.  Only use relavant {topic} questions that work without any context"
                            }
                        }
                    }
                },
            },
        }
    }
}

def create_data_creator_prompt(page_name: str, chunk: str, notes: str):
    return {"role": "user", "content": f"""Please call the function "create_data" to expertly build the highest quality large language model training data on the topic of {topic} for the following content titled {page_name}.
Ignore content not relevant to the main topic or any related topics.
Using the create_data function create an exhaustive list of prompt/response pairs, ensuring they are relevant and informative.
The questions should be posed from the viewpoint of someone unfamiliar with the topic, and answers should be detailed and easily understandable.
Include any relevant information not mentioned in the transcript if you are certain of its accuracy.
Do not use numbered lists in your responses.  Make sure to make it as robust as possible and try to get up to 500 words in the response
For instance if we are referring to an option in a window, make sure to include the name of the window or object we are setting properties on:
```transcript\n{chunk}\n```\nNotes: {notes}"""
        }

def create_data_creator_system_prompt(topic: str):
    return {"role": "system", "content": f"""You are an expert in {topic} and any related topics. 
You specialize in creating datasets of question/answer and instruction/task completion, and prompt / response pairs that provide the helpful assistance required when seeking information from the provided transcripts. 
Your expertise allows you to contextualize each pair of training data, focusing on the nuances of the topic and related areas.  
When working with transcripts and tutorials, you will never refer to the tutorial, its included projects, any projects whatsoever, author names, the contents title, or the human speaking in the transcipt.  
When writing an instruction/task completion, you will still use the question and answer fields in the responses function call even if its an instruction and not a question.
We are creating the most effective large language model training data possible, so its important that these question/answer pairs contains as robust data as possible.  Up to 500 words total and a minimum of 250 words.  
"""}