# Imports
import os
import sys
import torch
import random
import typing
import pydantic
import argparse
import bittensor as bt

bt.debug()

# Import protocol
current_script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
from protocol import TextToImage

# Load the config.
parser = argparse.ArgumentParser()
parser.add_argument( '--netuid', type = int )
bt.wallet.add_args( parser )
bt.subtensor.add_args( parser )
config = bt.config( parser )

# Instantiate the bittensor objects.
wallet = bt.wallet( config = config )
sub = bt.subtensor( config = config )
meta = bt.metagraph( config.netuid )
dend = bt.dendrite( wallet = wallet )

# For cosine similarity.
from sklearn.metrics.pairwise import cosine_similarity

# For image to text generation.
# Load roberta model for sentence similarity.
from transformers import AutoTokenizer, RobertaModel
text_to_embedding_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
text_to_embedding_model = RobertaModel.from_pretrained("roberta-base")

# Load Image captioning.
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
image_to_text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_to_text_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load prompt dataset.
from datasets import load_dataset
dataset = load_dataset("poloclub/diffusiondb")

# For prompt generation
from transformers import pipeline
prompt_generation_pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")

# Init the validator weights.
alpha = 0.01
weights = torch.rand_like( meta.uids, dtype = torch.float32 )

while True:

    # Get next uid to query.
    uid_to_query = random.choice( meta.uids.tolist() )

    # Get UID endpoint information.
    axon_to_query = meta.axon[ uid_to_query ]

    # Generate a random synthetic prompt.
    prompt = prompt_generation_pipe( next(dataset) )

    # Get response from endpoint.
    response = dend.query( axon_to_query, TextToImage( text = prompt ) )

    # Lets get the image caption.
    pixel_values = response.image.pixel_values
    generated_ids = image_to_text_model.generate(pixel_values)
    generated_text = image_to_text_tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]

    # Lets get the cosine similarity between the ask and the response.
    generate_text_inputs = text_to_embedding_tokenizer( generated_text, return_tensors="pt" )
    generated_embedding = text_to_embedding_model(**generate_text_inputs)
    generated_embedding_numpy = generated_embedding.last_hidden_state[-1][-1].reshape( (1,-1) ).detach().numpy()

    # Lets get the cosine similarity between the ask and the response.
    prompt_inputs = text_to_embedding_tokenizer( prompt, return_tensors="pt" )
    prompt_embedding = text_to_embedding_model(**prompt_inputs)
    prompt_embedding_numpy = prompt_embedding.last_hidden_state[-1][-1].reshape( (1,-1) ).detach().numpy()

    # Get cosine similarity
    next_weight_for_uid = cosine_similarity( generated_embedding_numpy, prompt_embedding_numpy )

    # Adjust the moving average
    weights[ uid_to_query ] =  ( 1 - alpha ) * weights[ uid_to_query ] + alpha * next_weight_for_uid

    # Optionally set weights
    current_block = sub.block
    if current_block % 100 == 0:
        uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids = meta.uids,
            weights = weights,
            netuid = config.netuid
        )
        sub.set_weights(
            wallet = wallet,
            netuid = config.netuid,
            weights = processed_weights,
            uids = uids,
        )
