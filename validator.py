# Imports
import os
import sys
import torch
import random
import typing
import pydantic
import argparse
import bittensor as bt
import numpy as np

import torchvision.transforms as transforms
from dendrite import AsyncDendritePool
from typing import List

import asyncio
from time import sleep

bt.trace()

# Import protocol
current_script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
from protocol import TextToImage

# Load the config.
parser = argparse.ArgumentParser()
parser.add_argument( '--netuid', type = int, default = 64 )
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://test.finney.opentensor.ai')
parser.add_argument('--subtensor._mock', type=bool, default=False)
bt.wallet.add_args( parser )
bt.subtensor.add_args( parser )
config = bt.config( parser )

# Instantiate the bittensor objects.
wallet = bt.wallet( config = config )
sub = bt.subtensor( config = config )
meta = sub.metagraph( config.netuid )
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
dataset = iter(load_dataset("poloclub/diffusiondb")['train'].shuffle().to_iterable_dataset())

# For prompt generation
from transformers import pipeline
prompt_generation_pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")

# Form the dendrite pool.
dendrite_pool = AsyncDendritePool( wallet = wallet, metagraph = meta )


# Init the validator weights.
alpha = 0.01
weights = torch.rand_like( meta.uids, dtype = torch.float32 )

 # Amount of images
num_images = 1
dendrites_per_query = 25

# Determine the rewards based on how close an image aligns to its prompt.
def calculate_rewards_for_prompt_alignment(query: TextToImage, responses: List[ TextToImage ]) -> torch.FloatTensor:

    # Takes the original query and a list of responses, returns a tensor of rewards equal to the length of the responses.
    init_scores = torch.zeros( len(responses), dtype = torch.float32 )

    for i, response in enumerate(responses):

        # if theres no images, skip this response.
        if len(response.images) == 0:
            print("No images in response", i, "skipping")
            continue

        # Lets get the cosine similarity between the ask and the response.
        prompt_inputs = text_to_embedding_tokenizer( query.text, return_tensors="pt" )
        prompt_embedding = text_to_embedding_model(**prompt_inputs)
        prompt_embedding_numpy = prompt_embedding.last_hidden_state[-1][-1].reshape( (1,-1) ).detach().numpy()

        img_scores = torch.zeros( num_images, dtype = torch.float32 )

        bt.logging.trace(f"Processing response {i} of {len(responses)}")

        for j, tensor_image in enumerate(response.images):
            # Lets get the image.
            image = transforms.ToPILImage()( bt.Tensor.deserialize(tensor_image) )
            # Lets get the image caption.
            pixel_values = image_processor(image, return_tensors='pt').pixel_values
            generated_ids = image_to_text_model.generate(pixel_values)
            generated_text = image_to_text_tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]

            # Lets get the cosine similarity between the ask and the response.
            generate_text_inputs = text_to_embedding_tokenizer( generated_text, return_tensors="pt" )
            generated_embedding = text_to_embedding_model(**generate_text_inputs)
            generated_embedding_numpy = generated_embedding.last_hidden_state[-1][-1].reshape( (1,-1) ).detach().numpy()
            
            # Get cosine similarity
            cosinesim = torch.tensor(cosine_similarity( generated_embedding_numpy, prompt_embedding_numpy ))

            img_scores[j] = cosinesim

        print(img_scores, i, "img_scores")
        
        # Get the average weight for the uid from _weights.
        init_scores[i] = torch.mean( img_scores )

    print(init_scores)

    # if sum is 0 then return empty vector
    if torch.sum( init_scores ) == 0:
        return init_scores

    # normalize the scores such that they sum to 1.
    init_scores = init_scores / torch.sum( init_scores )

    return init_scores

async def main():
    global weights
    # every 10 blocks, sync the metagraph.
    if sub.block % 10 == 0:
        meta.sync(subtensor = sub, )

    uids = meta.uids.tolist() 

    # if uids is longer than weight matrix, then we need to add more weights.
    if len(uids) > len(weights):
        bt.logging.trace("Adding more weights")
        size_difference = len(uids) - len(weights)
        new_weights = torch.zeros( size_difference, dtype = torch.float32 )
        # the new weights should be 0.8 * the average of all non 0 weights
        new_weights = new_weights + 0.8 * torch.mean( weights[weights != 0] )
        weights = torch.cat( (weights, new_weights) )
        del new_weights

    print('uids')
    print(uids)

    # Select up to dendrites_per_query random dendrites.
    dendrites_to_query = random.sample( uids, min( dendrites_per_query, len(uids) ) )

    # Generate a random synthetic prompt.
    prompt = prompt_generation_pipe( next(dataset)['prompt'] )[0]['generated_text']

    # Create the query.
    query = TextToImage(
        text = prompt,
        num_images_per_prompt = num_images,
        height = 512,
        width = 512,
        negative_prompt = "",
    )

    bt.logging.trace("Calling dendrite pool")
    bt.logging.trace(f"Query: {query.text}")
    bt.logging.trace("Dendrites:")
    bt.logging.trace(dendrites_to_query)

    # Get response from endpoint.
    responses = await dendrite_pool.async_forward(
        uids = dendrites_to_query,
        query = query,
        timeout = 30
    )
    bt.logging.trace("responses:")
    bt.logging.trace(responses)

    rewards: torch.FloatTensor = calculate_rewards_for_prompt_alignment( query, responses )
    bt.logging.trace("Rewards:")
    bt.logging.trace(rewards)
    # loop rewards and set weights.
    for i, uid in enumerate(dendrites_to_query):
        weights[uid] = weights[uid] + alpha * (rewards[i] or 0)

    # Normalize weights.
    weights = weights / torch.sum( weights )
    bt.logging.trace("Weights:")
    bt.logging.trace(weights)

    # Optionally set weights
    current_block = sub.block
    if current_block % 100 == 0:
        bt.logging.trace(f"Setting weights")
        uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids = meta.uids,
            weights = weights,
            netuid = config.netuid,
            subtensor = sub,
        )
        sub.set_weights(
            wallet = wallet,
            netuid = config.netuid,
            weights = processed_weights,
            uids = uids,
        )


while True:
     # wait for main to finish
    asyncio.run( main() )
