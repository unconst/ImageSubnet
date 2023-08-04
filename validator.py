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

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def load_and_preprocess_image_array(image_array, target_size):
    image_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocessed_images = []
    for image in image_array:
        if(image is None):
            preprocessed_images.append(None)
            continue
        image = image_transform(image).unsqueeze(0)
        preprocessed_images.append(image)

    return torch.cat(preprocessed_images, dim=0)

def extract_style_vectors(image_array, target_size=(224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = models.vgg19(pretrained=True).features.to(device)
    _model = nn.Sequential(*list(_model.children())[:35])

    images = load_and_preprocess_image_array(image_array, target_size).to(device)
    
    with torch.no_grad():
        style_vectors = _model(images)
    style_vectors = style_vectors.view(style_vectors.size(0), -1)
    return style_vectors

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity.item()

def compare_to_set(image_array, target_size=(224, 224)):
    # convert image array to index, image tuple pairs
    image_array = [(i, image) for i, image in enumerate(image_array)]
    # only process images that are not None
    style_vectors = extract_style_vectors([image for _, image in image_array if image is not None], target_size)
    # add back in the None images as zero vectors
    for i, image in image_array:
        if image is None:
            # style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)), style_vectors[i:]))
            # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_CUDA_cat)
            # Fixed version:
            style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)).to(style_vectors.device), style_vectors[i:]))

    similarity_matrix = torch.zeros(len(image_array), len(image_array))
    for i in range(style_vectors.size(0)):
        for j in range(style_vectors.size(0)):
            if image_array[i] is not None and image_array[j] is not None:
                similarity = cosine_similarity(style_vectors[i], style_vectors[j])
                likeness = 1.0 - similarity  # Invert the likeness to get dissimilarity
                likeness = min(1,max(0, likeness))  # Clip the likeness to [0,1]
                if likeness < 0.01:
                    likeness = 0
                similarity_matrix[i][j] = likeness

    return similarity_matrix.tolist()

def calculate_mean_dissimilarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [dissimilarity_matrix[i][j] for j in range(num_images) if i != j]
        # error: list index out of range
        if len(dissimilarity_values) == 0 or sum(dissimilarity_values) == 0:
            mean_dissimilarities.append(0)
            continue
        # divide by amount of non zero values
        non_zero_values = [value for value in dissimilarity_values if value != 0]
        mean_dissimilarity = sum(dissimilarity_values) / len(non_zero_values)
        mean_dissimilarities.append(mean_dissimilarity)

     # Min-max normalization
    non_zero_values = [value for value in mean_dissimilarities if value != 0]
    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [(value - min_value) / range_value for value in mean_dissimilarities]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [min(1,max(0, value)) for value in mean_dissimilarities]

    # Ensure sum of values is 1 (normalize)
    # sum_values = sum(mean_dissimilarities)
    # if sum_values != 0:
    #     mean_dissimilarities = [value / sum_values for value in mean_dissimilarities]

    return mean_dissimilarities

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
weights = torch.ones_like( meta.uids , dtype = torch.float32 ) * 0.5

 # Amount of images
num_images = 1
dendrites_per_query = 25

# Determine the rewards based on how close an image aligns to its prompt.
def calculate_rewards_for_prompt_alignment(query: TextToImage, responses: List[ TextToImage ]) -> (torch.FloatTensor, List[ Image.Image ]):

    # Takes the original query and a list of responses, returns a tensor of rewards equal to the length of the responses.
    init_scores = torch.zeros( len(responses), dtype = torch.float32 )
    top_images = []

    for i, response in enumerate(responses):

        # if theres no images, skip this response.
        if len(response.images) == 0:
            top_images.append(None)
            continue

        img_scores = torch.zeros( num_images, dtype = torch.float32 )

        with torch.no_grad():

            images = []

            for j, tensor_image in enumerate(response.images):
                # Lets get the image.
                image = transforms.ToPILImage()( bt.Tensor.deserialize(tensor_image) )

                images.append(image)
            
            ranking, scores = scoring_model.inference_rank(query.text, images)
            img_scores = torch.tensor(scores)
            # push top image to images (i, image)
            if len(images) > 1:
                top_images.append(images[ranking[0]-1])
            else:
                top_images.append(images[0])

        
        # Get the average weight for the uid from _weights.
        init_scores[i] = torch.mean( img_scores )
        
    # if sum is 0 then return empty vector
    if torch.sum( init_scores ) == 0:
        return torch.zeros( len(responses), dtype = torch.float32 )

    # preform exp on all values
    init_scores = torch.exp( init_scores )

    # set all values of 1 to 0
    init_scores[init_scores == 1] = 0

    # normalize the scores such that they sum to 1 but skip scores that are 0
    init_scores = init_scores / torch.sum( init_scores )


    return (init_scores, top_images)

def calculate_dissimilarity_rewards( images: List[ Image.Image ] ) -> torch.FloatTensor:
    # Takes a list of images, returns a tensor of rewards equal to the length of the images.
    init_scores = torch.zeros( len(images), dtype = torch.float32 )

    # Calculate the dissimilarity matrix.
    dissimilarity_matrix = compare_to_set(images)

    # Calculate the mean dissimilarity for each image.
    mean_dissimilarities = calculate_mean_dissimilarity(dissimilarity_matrix)

    # Calculate the rewards.
    for i, image in enumerate(images):
        init_scores[i] = mean_dissimilarities[i]

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

    bt.logging.trace('uids')
    bt.logging.trace(uids)

    # Select up to dendrites_per_query random dendrites.
    dendrites_to_query = random.sample( uids, min( dendrites_per_query, len(uids) ) )
    # dendrites_to_query = uids

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

    (rewards, best_images) = calculate_rewards_for_prompt_alignment( query, responses )
    dissimilarity_rewards: torch.FloatTensor = calculate_dissimilarity_rewards( best_images )

    # Calculate the final rewards.
    dissimilarity_weight = 0.15
    rewards = (1 - dissimilarity_weight) * rewards + dissimilarity_weight * dissimilarity_rewards    
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
    if current_block % 50 == 0:
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
