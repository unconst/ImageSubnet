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
import datetime

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
# Load the scoring model
import ImageReward as RM
scoring_model = RM.load("ImageReward-v1.0")

# Load prompt dataset.
from datasets import load_dataset
# generate random seed
seed=random.randint(0, 1000000)
dataset = iter(load_dataset("poloclub/diffusiondb")['train'].shuffle(seed=seed).to_iterable_dataset())

# For prompt generation
from transformers import pipeline
prompt_generation_pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")

# Form the dendrite pool.
dendrite_pool = AsyncDendritePool( wallet = wallet, metagraph = meta )


# Init the validator weights.
alpha = 0.01
# weights = torch.rand_like( meta.uids, dtype = torch.float32 )
weights = torch.ones_like( meta.uids, dtype = torch.float32 ) * 0.5

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

        img_scores = torch.zeros( num_images, dtype = torch.float32 )

        bt.logging.trace(f"Processing response {i} of {len(responses)}")

        with torch.no_grad():

            images = []

            for j, tensor_image in enumerate(response.images):
                # Lets get the image.
                image = transforms.ToPILImage()( bt.Tensor.deserialize(tensor_image) )

                images.append(image)
            
            ranking, scores = scoring_model.inference_rank(query.text, images)
            img_scores = torch.tensor(scores)

        
        # Get the average weight for the uid from _weights.
        init_scores[i] = torch.mean( img_scores )

    print(init_scores)

    # if sum is 0 then return empty vector
    if torch.sum( init_scores ) == 0:
        return torch.zeros( len(responses), dtype = torch.float32 )

    # preform exp on all values
    init_scores = torch.exp( init_scores )

    # set all values of 1 to 0
    init_scores[init_scores == 1] = 0

    # normalize the scores such that they sum to 1 but skip scores that are 0
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
    # dendrites_to_query = random.sample( uids, min( dendrites_per_query, len(uids) ) )
    dendrites_to_query = uids

    # Generate a random synthetic prompt.
    prompt = prompt_generation_pipe( next(dataset)['prompt'] )[0]['generated_text']

    # Create the query.
    query = TextToImage(
        text = prompt,
        num_images_per_prompt = num_images,
        height = 512,
        width = 512,
        negative_prompt = "",
        num_inference_steps = 50,
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

    # save all images in responses to a folder
    # imgid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # for i, response in enumerate(responses):
    #     for j, image in enumerate(response.images):
    #         # Lets get the image.
    #         image = transforms.ToPILImage()( bt.Tensor.deserialize(image) )
    #         image.save(f"images/{imgid}_{i}_{j}.png")

    rewards: torch.FloatTensor = calculate_rewards_for_prompt_alignment( query, responses )
    
    bt.logging.trace("Rewards:")
    bt.logging.trace(rewards)

    # reorder rewards to match dendrites_to_query
    _rewards = torch.zeros( len(uids), dtype = torch.float32 )
    for i, uid in enumerate(dendrites_to_query):
        _rewards[uids.index(uid)] = rewards[i]
    rewards = _rewards
    
    weights = weights + alpha * rewards


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
