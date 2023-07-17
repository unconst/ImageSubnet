# Imports
import os
import sys
import torch
import random
import typing
import pydantic
import argparse
import bittensor as bt

bt.trace()

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
meta = sub.metagraph( config.netuid )
dend = bt.dendrite( wallet = wallet )
print(meta.uids)

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
dataset = iter(load_dataset("poloclub/diffusiondb")['train'].to_iterable_dataset())

# For prompt generation
from transformers import pipeline
prompt_generation_pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")

# Init the validator weights.
alpha = 0.01
weights = torch.rand_like( meta.uids, dtype = torch.float32 )

 # Amount of images
num_images = 1

while True:

    uids = meta.uids.tolist() 
    # Get next uid to query.
    uid_to_query = random.choice( uids )
    

    # Get UID endpoint information.
    axon_to_query = meta.axons[ 0 ]

    # Generate a random synthetic prompt.
    prompt = prompt_generation_pipe( next(dataset)['prompt'] )[0]['generated_text']

    bt.logging.trace(f"Querying {uid_to_query} with prompt: {prompt}")

    # Get response from endpoint.
    response = dend.query( axon_to_query, TextToImage( text = prompt, num_images_per_prompt = num_images, negative_prompt="" ) )

    if(response == None):
        bt.logging.trace(f"Got no response from {uid_to_query}")
        continue

    if(response.images == None or len(response.images) == 0):
        bt.logging.trace(f"Got no images from {uid_to_query}")
        continue

    # slice the images requested from the response
    images = response.images[:num_images]
    bt.logging.trace(f"Got {len(images)} images from {uid_to_query}")

    weights = []

    for image in images:
        # Lets get the image caption.
        pixel_values = image.pixel_values
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
        weight_for_image = cosine_similarity( generated_embedding_numpy, prompt_embedding_numpy )
        weights.append( weight_for_image )
    
    # Get the average weight for the uid.
    next_weight_for_uid = torch.mean( torch.tensor( weights ) )
    bt.logging.trace(f"Got average weight {next_weight_for_uid} for {uid_to_query}")

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
