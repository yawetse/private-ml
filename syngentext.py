# %% [markdown]
# # Synthetic Transcript Generation using Microsoft Private Evolution (DPSDA)
#
# This notebook demonstrates how to use the DPSDA library to generate synthetic call transcripts based on a real dataset provided in a CSV file. It utilizes OpenAI's GPT model for generation while incorporating differential privacy mechanisms.
#
# **Prerequisites:**
# 1.  Install necessary libraries: `pip install dpsda pandas python-dotenv sentence-transformers`
# 2.  Create a `.env` file in the same directory as this notebook with your OpenAI API key:
#    ```
#    OPENAI_API_KEY=your_openai_api_key
#    ```
# 3.  Ensure your transcript CSV file (e.g., `your_transcripts.csv`) is accessible from this notebook's location.
# 4.  Download the prompt files (`random_api_prompt.json`, `variation_api_prompt.json`) from the [DPSDA GitHub example directory](https://github.com/microsoft/DPSDA/tree/main/example/text/yelp_openai) and place them in the same directory as this notebook.

# %%
# Import necessary libraries
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

# Import DPSDA components
from pe.data.text import TextData
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import OpenAILLM
from pe.embedding.text import SentenceTransformer
from pe.histogram import NearestNeighbors
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint

# Configure pandas behavior
pd.options.mode.copy_on_write = True

# %% [markdown]
# ## Configuration
# Set up paths, load environment variables (including the OpenAI API key), and configure logging.

# %%
# Define the output folder for results
exp_folder = "results/text/synthetic_transcripts"
# Get the directory of the current script/notebook
current_folder = os.path.dirname(os.path.abspath("__file__")) # Use __file__ for script context; adjust if running interactively without saving

# Create the results directory if it doesn't exist
os.makedirs(exp_folder, exist_ok=True)
os.makedirs(os.path.join(exp_folder, "synthetic_text"), exist_ok=True) # Create subfolder for CSV output

# Load environment variables from .env file
load_dotenv()

# Setup logging to file and console
setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

# %% [markdown]
# ## Load Data
# Load the call transcripts from the specified CSV file.

# %%
# Specify the path to your CSV file
csv_file_path = 'your_transcripts.csv' # <<< CHANGE THIS TO YOUR CSV FILENAME

# Load the CSV using pandas
try:
    df = pd.read_csv(csv_file_path)
    # Extract the 'TranscriptText' column into a list
    real_texts = df['TranscriptText'].tolist()
    print(f"Successfully loaded {len(real_texts)} transcripts from {csv_file_path}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}. Please check the path.")
    # Exit or handle error appropriately in a real script
    real_texts = [] # Assign empty list to avoid downstream errors in notebook context
except KeyError:
    print(f"Error: 'TranscriptText' column not found in {csv_file_path}. Please check the column name.")
    real_texts = []

# Create a DPSDA TextData object
if real_texts:
    data = TextData(texts=real_texts)
    num_private_samples = len(data.data_frame)
    print(f"Created TextData object with {num_private_samples} samples.")
else:
    print("Cannot proceed without loaded data.")
    # In a real scenario, you might stop execution here.

# %% [markdown]
# ## Initialize DPSDA Components
# Configure and initialize the components needed for the Private Evolution process:
# * **LLM:** The language model interface (OpenAI GPT-4o-mini).
# * **API:** The text augmentation API using the LLM.
# * **Embedding:** A sentence transformer model for text representation.
# * **Histogram:** The mechanism for density estimation (Nearest Neighbors).
# * **Population:** Manages the synthetic population during evolution.

# %%
# Check if data was loaded successfully before proceeding
if real_texts:
    # Configure the OpenAI LLM
    # Model: gpt-4o-mini (as used in the example script update)
    # Temperature: Controls randomness (higher means more random)
    # Num_threads: Parallel API calls
    llm = OpenAILLM(
        max_completion_tokens=128,
        model="gpt-4o-mini-2024-07-18",
        temperature=1.4,
        num_threads=4 # Adjust based on your OpenAI rate limits and CPU
    )
    print("Initialized OpenAILLM.")

    # Configure the LLM Augmentation API
    # Requires prompt template files (download from DPSDA repo)
    random_prompt_path = os.path.join(current_folder, "random_api_prompt.json")
    variation_prompt_path = os.path.join(current_folder, "variation_api_prompt.json")

    if not os.path.exists(random_prompt_path) or not os.path.exists(variation_prompt_path):
         print(f"Error: Prompt files not found. Make sure 'random_api_prompt.json' and 'variation_api_prompt.json' are in the directory: {current_folder}")
         # Handle error appropriately
    else:
        api = LLMAugPE(
            llm=llm,
            random_api_prompt_file=random_prompt_path,
            variation_api_prompt_file=variation_prompt_path,
            min_word_count=25, # Minimum words for generated text
            word_count_std=20, # Standard deviation for word count target
            token_to_word_ratio=1.2, # Estimated ratio for token calculation
            max_completion_tokens_limit=1200, # Limit for LLM generation
            blank_probabilities=0.5, # Probability for blanking parts of prompts
        )
        print("Initialized LLMAugPE.")

        # Configure the Sentence Transformer for embeddings
        # 'stsb-roberta-base-v2' is a common choice for semantic similarity
        embedding = SentenceTransformer(model="stsb-roberta-base-v2")
        print("Initialized SentenceTransformer embedding.")

        # Configure the Nearest Neighbors histogram
        histogram = NearestNeighbors(
            embedding=embedding,
            mode="L2", # Use L2 distance (Euclidean)
            lookahead_degree=0, # Parameter for histogram construction
        )
        print("Initialized NearestNeighbors histogram.")

        # Configure the PE Population
        population = PEPopulation(
            api=api,
            initial_variation_api_fold=3, # Folds for initial variations
            next_variation_api_fold=3, # Folds for subsequent variations
            keep_selected=True, # Keep selected samples in population
            selection_mode="rank" # Selection strategy
        )
        print("Initialized PEPopulation.")

# %% [markdown]
# ## Setup Callbacks and Loggers
# Configure components that run during or after the PE process:
# * **SaveTextToCSV:** Saves the generated synthetic text to CSV files.
# * **CSVPrint/LogPrint:** Log progress and metrics.

# %%
if real_texts:
    # Callback to save synthetic data periodically (or at the end)
    save_text_to_csv = SaveTextToCSV(
        output_folder=os.path.join(exp_folder, "synthetic_text")
    )
    print("Configured SaveTextToCSV callback.")

    # Loggers for console and CSV output
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()
    print("Configured loggers.")

# %% [markdown]
# ## Run Private Evolution
# Initialize the main `PE` runner and execute the generation process.
# * `num_samples_schedule`: Defines how many synthetic samples to generate. Here, we generate 5000 in one go.
# * `delta`: Differential privacy parameter, often set based on the dataset size.
# * `epsilon`: Differential privacy parameter (privacy budget).

# %%
if real_texts and 'data' in locals() and 'population' in locals() and 'histogram' in locals():
    # Calculate delta based on the number of private samples
    # This is a common heuristic for setting delta in DP
    delta = 1.0 / num_private_samples / np.log(num_private_samples)
    print(f"Calculated delta: {delta}")

    # Initialize the PE runner
    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_text_to_csv], # Add other callbacks if needed (e.g., ComputeFID)
        loggers=[csv_print, log_print],
    )
    print("Initialized PE runner.")

    # Run the PE process
    print("Starting Private Evolution process to generate 5000 samples...")
    pe_runner.run(
        num_samples_schedule=[5000], # Generate 5000 samples in one generation step
        delta=delta,
        epsilon=1.0, # Set your desired privacy budget (epsilon)
        checkpoint_path=None, # Disable checkpointing for this simple run
    )
    print(f"Finished PE process. Synthetic data saved in: {os.path.join(exp_folder, 'synthetic_text')}")
else:
    print("Skipping PE run due to errors in previous steps (data loading or component initialization).")

