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
import imagehash
import time

import torchvision.transforms as transforms
from dendrite import AsyncDendritePool
from typing import List, Tuple

import asyncio
from time import sleep

bt.trace()
# Import protocol
current_script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
from db import conn, delete_prompts_by_uid, delete_prompts_by_timestamp, create_or_get_hash_id, create_prompt, create_batch, get_prompts_of_random_batch, Prompt
from protocol import TextToImage, ImageToImage, validate_synapse, ValidatorSettings
from utils import check_for_updates, __version__
check_for_updates()

# Load the config.
parser = argparse.ArgumentParser()
parser.add_argument( '--netuid', type = int, default = 5 )
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://entrypoint-finney.opentensor.ai')
parser.add_argument('--subtensor._mock', type=bool, default=False)
parser.add_argument('--validator.allow_nsfw', type=bool, default=False)
parser.add_argument('--validator.save_dir', type=str, default='./images')
parser.add_argument('--validator.save_images', type=bool, default=False)
parser.add_argument('--validator.use_absolute_size', type=bool, default=False) # Set to True if you want to 100% match the input size, else just match the aspect ratio
parser.add_argument('--validator.label_images', type=bool, default=False, help="if true, label images with dendrite uid and score")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--axon.port', type=int, default=3000)
bt.wallet.add_args( parser )
bt.subtensor.add_args( parser )
config = bt.config( parser )

# if save dir different than default, save_images should be true
if config.validator.save_dir != './images':
    config.validator.save_images = True

# if save dir doesnt exist, create it
if not os.path.exists(config.validator.save_dir) and config.validator.save_images:
    bt.logging.trace("Save directory doesnt exist, creating it")
    os.makedirs(config.validator.save_dir)

# Instantiate the bittensor objects.
wallet = bt.wallet( config = config )
sub = bt.subtensor( config = config )
meta = sub.metagraph( config.netuid )
meta.sync( subtensor = sub )
dend = bt.dendrite( wallet = wallet )

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from utils import StableDiffusionSafetyChecker, transform
from transformers import CLIPImageProcessor
from fabric.utils import get_free_gpu, tile_images
import matplotlib.font_manager as fm


DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")

# For image to text generation.
# Load the scoring model
import ImageReward as RM
scoring_model = RM.load("ImageReward-v1.0", device=DEVICE)

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

# list of sizes
sizes = [512, 768, 1024, 1536]

# list of aspect ratios [(1, 1), (4, 3), (16, 9)]
aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]


# Init the validator weights.
alpha = 0.0001
# weights = torch.rand_like( meta.uids, dtype = torch.float32 )
weights = torch.ones_like( meta.uids , dtype = torch.float32 )

# multiply weights by the active tensor
curr_block = sub.block

# loop over all last_update, any that are within 600 blocks are set to 1 others are set to 0 
weights = weights * meta.last_update > curr_block - 600

# all nodes with more than 1e3 total stake are set to 0 (sets validtors weights to 0)
weights = weights * (meta.total_stake < 1.024e3) 

# set all nodes without ips set to 0
weights = weights * torch.Tensor([meta.neurons[uid].axon_info.ip != '0.0.0.0' for uid in meta.uids]) * 0.5

# normalize
weights = weights / torch.sum(weights)
_last_normalized_weights = curr_block - (curr_block % 25)

 # Amount of images
num_images = 1
total_dendrites_per_query = 25
minimum_dendrites_per_query = 3

last_updated_block = curr_block - (curr_block % 100)
last_reset_weights_block = curr_block
_loop = 0

safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( DEVICE )
processor = CLIPImageProcessor()

# create a dictionary to track the last time a uid was queried
last_queried = {}

async def main():
    global weights, last_updated_block, last_reset_weights_block, last_queried, _loop, _last_normalized_weights

    SyncMetagraphIfNeeded()
    uids = meta.uids.tolist() 

    # if uids is longer than weight matrix, then we need to add more weights.
    ExtendWeightMatrixIfNeeded(uids)

    ### SET WEIGHTS SECTION ###
    # Set weights was moved to the top of the function in case t2i or i2i returns early for multiple blocks causing weight setting to never happen
    
    current_block = sub.block
    if current_block - last_updated_block  >= 100:
        bt.logging.trace(f"Setting weights")

        # Normalize weights.
        weights = weights / torch.sum( weights )
        _last_normalized_weights = sub.block

        bt.logging.trace("Weights:")
        bt.logging.trace(weights)

        _has_set = False
        _retries = 0
        while _has_set == False:
            try:
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
                _has_set = True
            except Exception as e:
                _sleep_time = 2 ** _retries
                _sleep_time = 30 if _sleep_time > 30 else _sleep_time
                bt.logging.warning(f"Error setting weights: {e} retrying in {_sleep_time} seconds")
                sleep(_sleep_time)
                continue
        last_updated_block = current_block

        # delete_prompts_by_timestamp for timestamps older than 48h
        delete_prompts_by_timestamp(conn, time.time() - 172800)

        check_for_updates()
    elif sub.block - _last_normalized_weights >= 25:
        # Normalize weights.
        weights = weights / torch.sum( weights )
        _last_normalized_weights = sub.block
        bt.logging.trace("25 blocks have passed, normalizing weights")

    ### END SET WEIGHTS SECTION ###


    # use rand int to select int between 1-10
    randomint = random.randint(1, 10)
    if randomint == 1 and False: # Disabled for now
        # get batch between 48h ago and now
        prompts = get_prompts_of_random_batch(conn, time.time() - 172800)

        # if prompts is none, skip block
        if prompts is None:
            return


        # uids
        dendrites_to_query = [prompt.uid for prompt in prompts]
        bt.logging.trace(f"Querying {len(dendrites_to_query)} dendrites with requery")
        # if len of dend is 0 warn and skip block
        if len(dendrites_to_query) == 0:
            bt.logging.warning("No dendrites to query, skipping block")
            return

        rewards = torch.zeros( len(dendrites_to_query), dtype = torch.float32 )
        # create a dictionary where the key is the uid and the value is a list of prompts for that uid sorted in the order of image_order_Id
        prompts_dict = {}
        for prompt in prompts:
            if prompt.uid not in prompts_dict:
                prompts_dict[prompt.uid] = []
            prompts_dict[prompt.uid].append(prompt)
        for uid in prompts_dict:
            prompts_dict[uid] = sorted(prompts_dict[uid], key=lambda x: x.image_order_id)

        # get the maximum number of images for any uid
        prompts = [len(prompts_dict[uid]) for uid in prompts_dict]

        if len(prompts) == 0:
            bt.logging.warning("No prompts found, skipping block")
            return

        maximum_number_of_images = max(prompts)

        # recreate the query from the prompt
        query = TextToImage(
            text = prompts[0].prompt,
            num_images_per_prompt = maximum_number_of_images,
            height = prompts[0].height,
            width = prompts[0].width,
            negative_prompt = prompts[0].negative,
            nsfw_allowed=config.validator.allow_nsfw,
            seed=prompts[0].seed,
        )

        query, timeout, responses, dendrites_to_query = await AsyncQueryTextToImage(uids, query)

        hashes = GetImageHashesOfResponses(responses)

        # hashes is a 2d array where the first dimension corelates with the order of responses/uids queried, the second is the hash in the order of images supplied

        # loop through hashes and check to see if they match the hash of the original prompts object

        # loop through all the hashes
        for i, _hashes in enumerate(hashes):
            # loop through all the hashes for that uid
            uid = dendrites_to_query[i]
            for j, _hash in enumerate(_hashes):
                # if the hash matches the original prompt hash, set the reward to 0
                current_reward = rewards[dendrites_to_query.index(uid)]
                # check if j exists in prompts_dict[uid]
                if j < len(prompts_dict[uid]):
                    if _hash == prompts_dict[uid][j].hash_value:
                        if current_reward == 0:
                            rewards[dendrites_to_query.index(uid)] = 1
                    else:
                        rewards[dendrites_to_query.index(uid)] = -1
                else:
                    pass

        # set all rewards less than 0 to 0
        rewards[rewards < 0] = 0

        # if sum of rewards is 0, skip block
        if torch.sum( rewards ) == 0:
            weights = weights * 0.993094
            return
        
        # skip normalization

        # extend the rewards matrix out to the entire length of uids so it can be added into weights
        rewards = ExtendRewardMatrixToUidsLength(uids, dendrites_to_query, rewards)

        # because we don't normalize we need to add a scaling factor additional to that of alpha
        scaling = 0.075

        weights = weights + alpha * rewards * scaling

        # every loop scale weights by 0.993094, sets half life to 100 blocks
        weights = weights * 0.993094

    else:

        ### TEXT TO IMAGE SECTION ###

        _, prompt = GeneratePrompt()

        (width, height) = get_resolution()

        # Create the query.
        query = TextToImage(
            text = prompt,
            num_images_per_prompt = num_images,
            height = height,
            width = width,
            negative_prompt = "",
            nsfw_allowed=config.validator.allow_nsfw,
            seed=random.randint(0, 1e9)
        )

        batch_id = create_batch(conn, time.time())

        query, timeout, responses, dendrites_to_query = await AsyncQueryTextToImage(uids, query)

        rewards, hashes, best_pil_image, best_image_hash = ScoreTextToImage(responses, batch_id, query, dendrites_to_query)

        # if sum of rewards is 0, skip block
        if torch.sum( rewards ) == 0:
            bt.logging.trace("All rewards are 0, skipping block")
            weights = weights * 0.993094
            return

        # extend the rewards matrix out to the entire length of uids so it can be added into weights
        rewards = ExtendRewardMatrixToUidsLength(uids, dendrites_to_query, rewards)

        weights = weights + alpha * rewards

        prompt = query.text

        if best_pil_image is None:
            bt.logging.warning("No best image found in text to image batch, skipping image to image")
            weights = weights * 0.993094
            return

        ### END TEXT TO IMAGE SECTION ###



        ### IMAGE TO IMAGE SECTION ###

        similarities = ["low", "medium", "high"]

        serialized_best_image = bt.Tensor.serialize(transform(best_pil_image))

        # Create ImageToImage query
        i2i_query = ImageToImage(
            image = serialized_best_image,
            height = best_pil_image.height,
            width = best_pil_image.width,
            negative_prompt = "",
            # do a 5050 chance of using the prompt or just empty string
            text = prompt if random.randint(0, 1) == 0 else "",
            nsfw_allowed=config.validator.allow_nsfw,
            seed=random.randint(0, 1e9),
            similarity = similarities[random.randint(0, len(similarities)-1)]
        )

        batch_id = create_batch(conn, time.time())

        i2i_rewards, i2i_responses, dendrites_to_query = await AsyncQueryImageToImage(uids, i2i_query, prompt, best_image_hash, timeout, batch_id)

        # if sum of rewards is 0, skip block
        if torch.sum( i2i_rewards ) == 0 or torch.max( i2i_rewards ) == 0:
            weights = weights * 0.993094
            return

        i2i_rewards = i2i_rewards / torch.max(i2i_rewards)

        
        # loop through all images and remove black images
        SaveImages(dendrites_to_query, prompt, i2i_query, i2i_responses, i2i_rewards)

        # reorder rewards to match dendrites_to_query
        _rewards = torch.zeros( len(uids), dtype = torch.float32 )
        for i, uid in enumerate(dendrites_to_query):
            if not torch.isnan(i2i_rewards[i]):
                _rewards[uids.index(uid)] = i2i_rewards[i]
            else:
                bt.logging.warning(f"Reward for uid {uid} is nan (326)! This should not be the case!")
        i2i_rewards = _rewards
        
        weights = weights + alpha * i2i_rewards

        ### END IMAGE TO IMAGE SECTION ###


    ### WEIGHT MANAGEMENT SECTION ###

    # every loop scale weights by 0.993094, sets half life to 100 blocks
    weights = weights * 0.993094

    # hard set weights with 1024 stake to 0
    weights[meta.total_stake > 1.024e3] = 0

    # if weight is less than 1/2048, set it to 0
    weights[weights < 1/2048] = 0

    ### END WEIGHT MANAGEMENT SECTION ###

    _loop += 1
    bt.logging.trace(f"Finished with loop {_loop} at block {sub.block}, { 100 - (sub.block - last_updated_block) } blocks until weights are updated")

### END MAIN FUNCTION ###

async def AsyncQueryImageToImage(uids, i2i_query, prompt, best_image_hash, timeout, batch_id):
    queryable_uids, active_miners, dendrites_per_query = GetQueryableUids(uids)

    timeout_increase = GetTimeoutIncrease(active_miners, dendrites_per_query)

    dendrites_to_query = GetDendritesToQuery(uids, queryable_uids, dendrites_per_query)

    # Get response from endpoints
    i2i_responses = await dendrite_pool.async_forward(
        uids = dendrites_to_query,
        query = i2i_query,
        timeout = timeout * timeout_increase
    )

    SetDendritesLastQueried(dendrites_to_query)

    dendrites_to_query, i2i_responses = CheckForNSFW(dendrites_to_query, i2i_responses)

    i2i_rewards, _, _, _ = CalculateRewards(dendrites_to_query, batch_id, prompt, i2i_query, i2i_responses, best_image_hash)

    return i2i_rewards, i2i_responses, dendrites_to_query

async def AsyncQueryTextToImage(all_uids, query):
    global weights, last_updated_block, last_reset_weights_block, last_queried, _loop

    queryable_uids, active_miners, dendrites_per_query = GetQueryableUids(all_uids)

    timeout_increase = GetTimeoutIncrease(active_miners, dendrites_per_query)

    dendrites_to_query = GetDendritesToQuery(all_uids, queryable_uids, dendrites_per_query)

    # total pixels
    total_pixels = query.width * query.height

    base_timeout = 12
    base_timeout_size = 512*512

    max_timeout = 30

    
    # calculate timeout based on size of image, image size goes up quadraticly but timeout goes up linearly, so if you go from 512,512 -> 1024,1024, the timeout should be 3x
    timeout = CalculateTimeout(total_pixels, base_timeout, base_timeout_size, max_timeout)

    # Get response from endpoint.
    responses = await dendrite_pool.async_forward(
        uids = dendrites_to_query,
        query = query,
        timeout = timeout * timeout_increase
    )

    # for each queried uid, set the last queried time to now
    SetDendritesLastQueried(dendrites_to_query)

    return query, timeout, responses, dendrites_to_query

def ScoreTextToImage(responses, batch_id, query, dendrites_to_query):
    # validate all responses, if they fail validation remove both the response from responses and dendrites_to_query
    dendrites_to_query, responses = ValidateResponses(dendrites_to_query, responses)

    dendrites_to_query, responses = CheckForNSFW(dendrites_to_query, responses)

    rewards, hashes, best_pil_image,best_image_hash = CalculateRewards(dendrites_to_query, batch_id, query.text, query, responses)
    
    return rewards, hashes, best_pil_image,best_image_hash

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
    _model = models.vgg19(pretrained=True).features.to(DEVICE)
    _model = nn.Sequential(*list(_model.children())[:35])

    images = load_and_preprocess_image_array(image_array, target_size).to(DEVICE)
    
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

    # if there are no images, return an empty matrix
    if len(image_array) == 0:
        return []

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
    for i in range(len(image_array)):
        for j in range(len(image_array)):
            if image_array[i] is not None and image_array[j] is not None:
                if torch.sum(style_vectors[i]) == 0 or torch.sum(style_vectors[j]) == 0:
                    similarity_matrix[i][j] = 0
                else:
                    # Similarity score of 1 means the images are identical, 0 means they are completely different
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

    if(len(non_zero_values) == 0):
        return [0.5] * num_images

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

def get_resolution(size_index = None, aspect_ratio_index = None):
    # pick a random size and aspect ratio
    if size_index is None or size_index >= len(sizes):
        size_index = random.randint(0, len(sizes)-1)
    if aspect_ratio_index is None or aspect_ratio_index >= len(aspect_ratios):
        aspect_ratio_index = random.randint(0, len(aspect_ratios)-1)
    
    # get the size and aspect ratio
    size = sizes[size_index]
    aspect_ratio = aspect_ratios[aspect_ratio_index]

    # calculate the width and height
    width = size
    height = size
    # keep the largest side as size, and calculate the other side based on the aspect ratio
    if aspect_ratio[0] > aspect_ratio[1]:
        width = size
        height = int(size * aspect_ratio[1] / aspect_ratio[0])
    else:
        width = int(size * aspect_ratio[0] / aspect_ratio[1])
        height = size

    return (width, height)

# Determine the rewards based on how close an image aligns to its prompt.
def calculate_rewards_for_prompt_alignment(query: TextToImage, responses: List[ TextToImage ]) -> (torch.FloatTensor, List[ Image.Image ]):

    # Takes the original query and a list of responses, returns a tensor of rewards equal to the length of the responses.
    init_scores = torch.zeros( len(responses), dtype = torch.float32 )
    top_images = []

    print("Calculating rewards for prompt alignment")
    print(f"Query: {query.text}")
    print(f"Responses: {len(responses)}")

    for i, response in enumerate(responses):
        print(response, type(response))

        # if theres no images, skip this response.
        if len(response.images) == 0:
            top_images.append(None)
            continue

        img_scores = torch.zeros( num_images, dtype = torch.float32 )
        try:
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
        except Exception as e:
            print(e)
            print("error in " + str(i))
            print(response)
            top_images.append(None)
            continue

        
        # Get the average weight for the uid from _weights.
        init_scores[i] = torch.mean( img_scores )
        #  if score is < 0, set it to 0
        if init_scores[i] < 0:
            init_scores[i] = 0
        
    # if sum is 0 then return empty vector
    if torch.sum( init_scores ) == 0:
        return (init_scores, top_images)

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

    # If array is all nones, return 0 vector of length len(images)
    if all(image is None for image in images):
        return init_scores

    # Calculate the dissimilarity matrix.
    dissimilarity_matrix = compare_to_set(images)

    # Calculate the mean dissimilarity for each image.
    mean_dissimilarities = calculate_mean_dissimilarity(dissimilarity_matrix)

    # Calculate the rewards.
    for i, image in enumerate(images):
        init_scores[i] = mean_dissimilarities[i]

    return init_scores

def add_black_border(image, border_size):
    # Create a new image with the desired dimensions
    new_width = image.width
    new_height = image.height + border_size
    new_image = Image.new("RGB", (new_width, new_height), color="black")
    
    # Paste the original image onto the new image, leaving space at the top for the border
    new_image.paste(image, (0, border_size))
    
    return new_image

def ExtendRewardMatrixToUidsLength(all_uids, dendrites_to_query, rewards):
    _rewards = torch.zeros( len(all_uids), dtype = torch.float32 )
    for i, uid in enumerate(dendrites_to_query):
        # check if rewards[i] is nan
        if not torch.isnan(rewards[i]):
            _rewards[all_uids.index(uid)] = rewards[i]
        else:
            bt.logging.warning(f"Reward for uid {uid} is nan! This should not be the case!")
    rewards = _rewards
    return rewards

def CalculateRewards(dendrites_to_query, batch_id, prompt, query, responses, best_image_hash = None):
    (rewards, best_images) = calculate_rewards_for_prompt_alignment( query, responses )

    if torch.sum( rewards ) == 0:
        return rewards, [], None, None
    
    rewards = rewards / torch.max(rewards)

    dissimilarity_rewards: torch.FloatTensor = calculate_dissimilarity_rewards( best_images )

    # dissimilarity isnt the same length because we filtered out images with 0 reward, so we need to create a new tensor of length rewards
    new_dissimilarity_rewards = torch.zeros( len(rewards), dtype = torch.float32 )

    for i, reward in enumerate(rewards):
        if reward != 0:
            new_dissimilarity_rewards[i] = dissimilarity_rewards[i]

    dissimilarity_rewards = new_dissimilarity_rewards

    dissimilarity_rewards = dissimilarity_rewards / torch.max(dissimilarity_rewards)

    # my goal with dissimilarity_rewards is to encourage diversity in the images

    # normalize rewards such that the highest value is 1

    dissimilarity_weight = 0.15
    rewards = rewards + dissimilarity_weight * dissimilarity_rewards

    # Perform imagehash (perceptual hash) on all images. Any matching images are given a reward of 0.
    hash_rewards, hashes = ImageHashRewards(dendrites_to_query, responses, rewards)
    bt.logging.trace(f"Hash rewards: {hash_rewards}")
    
    # add hashes to the database
    for i, _hashes in enumerate(hashes):
        try:
            resp = responses[i] # TextToImage class
            uid = dendrites_to_query[i]
            for _hash in _hashes:
                hash_already_exists = create_prompt(conn, batch_id, _hash, uid, prompt, "", resp.seed, resp.height, resp.width, time.time(), best_image_hash)
                if hash_already_exists:
                    bt.logging.trace(f"Detected duplicate image from dendrite {dendrites_to_query[i]}")
                    hash_rewards[i] = 0
        except Exception as e:
            bt.logging.trace(f"Error in imagehash: {e}") if best_image_hash is None else bt.logging.trace(f"Error in i2i imagehash: {e}")
            print(e)
            pass
    
    # multiply rewards by hash rewards
    rewards = rewards * hash_rewards

    # get best image from rewards
    best_image_index = torch.argmax(rewards)
    best_pil_image = best_images[best_image_index]
    best_image_hash = hashes[best_image_index][0]

   

    if torch.sum( rewards ) == 0:
        return rewards, hashes, None, None
    
     # log uids
    bt.logging.trace(f"UIDs: {dendrites_to_query}")
    # log all rewards and the best image index / hash
    bt.logging.trace(f"Calculated Rewards: {rewards}")
    # log best score
    bt.logging.trace(f"Best score: {torch.max(rewards)} UID: {dendrites_to_query[best_image_index]} HASH: {best_image_hash}")

    rewards = rewards / torch.max(rewards)
    return rewards,hashes,best_pil_image,best_image_hash



def CheckForNSFW(dendrites_to_query, responses):
    if not config.validator.allow_nsfw:
        for i, response in enumerate(responses):
            # delete all none images
            for j, image in enumerate(response.images):
                if image is None:
                    del responses[i].images[j]
            if len(response.images) == 0:
                continue
            try:
                clip_input = processor([bt.Tensor.deserialize(image) for image in response.images], return_tensors="pt").to( DEVICE )
                images, has_nsfw_concept = safetychecker.forward(images=response.images, clip_input=clip_input.pixel_values.to( DEVICE ))

                any_nsfw = any(has_nsfw_concept)
                if any_nsfw:
                    bt.logging.trace(f"Detected NSFW image(s) from dendrite {dendrites_to_query[i]}")

                # remove all nsfw images from the response
                for j, has_nsfw in enumerate(has_nsfw_concept):
                    if has_nsfw:
                        del responses[i].images[j]
            except Exception as e:
                print(response.images)
                bt.logging.trace(f"Error in NSFW detection: {e}")
                pass
    
    return dendrites_to_query, responses

def ValidateResponses(dendrites_to_query, responses):
    for i, response in enumerate(responses):
        valid, error = validate_synapse(response)
        if not valid:
            bt.logging.trace(f"Detected invalid response from dendrite {dendrites_to_query[i]}: {error}")
            del responses[i]
            del dendrites_to_query[i]
    
    return dendrites_to_query, responses

def CalculateTimeout(total_pixels, base_timeout, base_timeout_size, max_timeout):
    if (total_pixels / base_timeout_size) > 1:
        timeout = base_timeout * (total_pixels / base_timeout_size) * 0.75
    else:
        timeout = base_timeout
    # if timeout is greater than max timeout, set it to max timeout
    if timeout > max_timeout:
        timeout = max_timeout

    # increase timeout for multiple images
    if (num_images > 1):
        timeout = timeout * (num_images*(2/3))
    return timeout

def GeneratePrompt():
    global dataset
    
    # Generate a random synthetic prompt. cut to first 20 characters.
    try:
        initial_prompt = next(dataset)['prompt']
    except:
        seed=random.randint(0, 1000000)
        dataset = iter(load_dataset("poloclub/diffusiondb")['train'].shuffle(seed=seed).to_iterable_dataset())
        initial_prompt = next(dataset)['prompt']
    # split on spaces
    initial_prompt = initial_prompt.split(' ')
    # pick a random number of words to keep
    keep = random.randint(1, len(initial_prompt))
    # max of 6 words
    keep = min(keep, 6)
    # keep the first keep words
    initial_prompt = ' '.join(initial_prompt[:keep])
    prompt = prompt_generation_pipe( initial_prompt, min_length=30 )[0]['generated_text']
    return initial_prompt,prompt

def ExtendWeightMatrixIfNeeded(uids):
    global weights
    if len(uids) > len(weights):
        bt.logging.trace("Adding more weights")
        size_difference = len(uids) - len(weights)
        new_weights = torch.zeros( size_difference, dtype = torch.float32 )
        # the new weights should be 0.3 * the median of all non 0 weights
        new_weights = new_weights + 0.3 * torch.median( weights[weights != 0] )
        weights = torch.cat( (weights, new_weights) )
        del new_weights

def SyncMetagraphIfNeeded():
    global sub, meta, weights

    # every 10 blocks, sync the metagraph.
    if sub.block % 10 == 0:
        # create old list of (uids, hotkey)
        old_uids = list(zip(meta.uids.tolist(), meta.hotkeys))
        _retries = 0
        _not_synced = True
        while _not_synced:
            try:
                meta.sync(subtensor = sub, )
                _not_synced = False
                # create new list of (uids, hotkey)
                new_uids = list(zip(meta.uids.tolist(), meta.hotkeys))
                # if the lists are different, reset weights for that uid
                for i in range(len(new_uids)):
                    if len(old_uids) > i:
                        if old_uids[i] != new_uids[i]:
                            weights[i] = 0.3 * torch.median( weights[weights != 0] )
                            
                            # delete all prompts for that uid
                            delete_prompts_by_uid(conn, new_uids[i][0])
                    else:
                        weights[i] = 0.3 * torch.median( weights[weights != 0] )
            except:
                _retries += 1
                _seconds_to_wait = 2 ** _retries
                if _seconds_to_wait > 30:
                    _seconds_to_wait = 30
                bt.logging.trace("Error in syncing metagraph... retrying in {} seconds".format(_seconds_to_wait))
                time.sleep(_seconds_to_wait)

def SetDendritesLastQueried(dendrites_to_query):
    global last_queried
    for uid in dendrites_to_query:
        last_queried[uid] = datetime.datetime.now()

    return last_queried

def GetTimeoutIncrease(active_miners, dendrites_per_query):
    timeout_increase = 1

    if dendrites_per_query > active_miners:
        bt.logging.warning(f"Warning: not enough active miners to sufficently validate images, rewards may be inaccurate. Active miners: {active_miners}, Minimum per query: {minimum_dendrites_per_query}")
    elif active_miners < dendrites_per_query * 3:
        bt.logging.warning(f"Warning: not enough active miners, miners may be overloaded from other validators. Enabling increased timeout.")
        timeout_increase = 2
    return timeout_increase

def GetDendritesToQuery(uids, queryable_uids, dendrites_per_query):
    # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
    zipped_uids = list(zip(uids, queryable_uids))
    filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
    dendrites_to_query = random.sample( filtered_uids, min( dendrites_per_query, len(filtered_uids) ) )
    return dendrites_to_query

def GetQueryableUids(uids):
    # Select up to dendrites_per_query random dendrites.
    queryable_uids = (meta.last_update > curr_block - 600) * (meta.total_stake < 1.024e3)

    # if queryable_uids doesnt match the length of meta.neurons, extend it

    # for all uids, check meta.neurons[uid].axon_info.ip == '0.0.0.0' if so, set queryable_uids[uid] to false
    queryable_uids = queryable_uids * torch.Tensor([meta.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids[:len(queryable_uids)]])

    # loop through queryable uids and check if if they have been queried in the last 2 minutes, if so, set queryable_uids[uid] to 0
    for uid in uids:
        if uid in last_queried:
            if (datetime.datetime.now() - last_queried[uid]).total_seconds() < 120:
                queryable_uids[uids.index(uid)] = 0

    active_miners = torch.sum(queryable_uids)
    dendrites_per_query = total_dendrites_per_query

    # if there are no active miners, set active_miners to 1
    if active_miners == 0:
        active_miners = 1

    # if there are less than dendrites_per_query * 3 active miners, set dendrites_per_query to active_miners / 3
    if active_miners < total_dendrites_per_query * 3:
        dendrites_per_query = int(active_miners / 3)
    else:
        dendrites_per_query = total_dendrites_per_query

    # less than 1 set to 1
    if dendrites_per_query < minimum_dendrites_per_query:
        dendrites_per_query = minimum_dendrites_per_query
    return queryable_uids,active_miners,dendrites_per_query

def get_system_fonts():
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    return font_list

# find DejaVu Sans font
if (config.validator.label_images == True):
    fonts = get_system_fonts()
    dejavu_font = None
    for font in fonts:
        if "DejaVu" in font:
            dejavu_font = font
            break 

    default_font = ImageFont.truetype(dejavu_font, 30)

def SaveImages(dendrites_to_query, prompt, query, responses, rewards):
    # if save images is true, save the images to a folder
    if config.validator.save_images == True:
        all_images_and_scores = []
        for i, response in enumerate(responses):
            images = response.images
            for j, image in enumerate(images):
                try:
                    img = bt.Tensor.deserialize(image)
                except:
                    bt.logging.trace(f"Detected invalid image to deserialize from dendrite {dendrites_to_query[i]}")
                    del responses[i].images[j]
                    continue
                if img.sum() == 0:
                    bt.logging.trace(f"Detected black image from dendrite {dendrites_to_query[i]}")
                    del responses[i].images[j]
                else:
                    # add the uid to the image in the top left with PIL
                    pil_img =  transforms.ToPILImage()( img )

                    # get size of image, if it doesnt match the size of the request, check to see if it matches the aspect ratio, if not delete it from responses
                    if pil_img.width != query.width or pil_img.height != query.height:
                        if config.validator.use_absolute_size:
                            bt.logging.trace(f"Detected image with incorrect size from dendrite {dendrites_to_query[i]}")
                            del responses[i].images[j]
                            continue
                        if pil_img.width / pil_img.height != query.width / query.height:
                            bt.logging.trace(f"Detected image with incorrect aspect ratio from dendrite {dendrites_to_query[i]}")
                            del responses[i].images[j]
                            continue

                    if (config.validator.label_images == True):
                        draw = ImageDraw.Draw(pil_img)
                        width = pil_img.width
                        
                        draw.text((5, 5), str(dendrites_to_query[i]), (255, 255, 255), font=default_font)
                        # draw score in top right
                        draw.text((width - 50, 5), str(round(rewards[i].item(), 3)), (255, 255, 255), font=default_font)
                        # downsize image in half
                        # pil_img = pil_img.resize( (int(pil_img.width / 2), int(pil_img.height / 2)) )
                    all_images_and_scores.append( (pil_img, rewards[i].item()) )

        
        # sort by score
        all_images_and_scores.sort(key=lambda x: x[1], reverse=True)
        # get the images
        all_images = [x[0] for x in all_images_and_scores]
        tiled_images = tile_images( all_images )
        # extend the image by 90 px on the top
        tiled_images = add_black_border( tiled_images, 90 )
        draw = ImageDraw.Draw(tiled_images)
        
        # add text in the top
        draw.text((10, 10), prompt.encode("utf-8", "ignore").decode("utf-8"), (255, 255, 255), font=default_font)

        # save the image
        tiled_images.save( f"images/{sub.block}.png", "PNG" )
        # save a text file with the prompt
        with open(f"images/{sub.block}.txt", "w") as f:
            f.write(prompt)

def ImageHashRewards(dendrites_to_query, responses, rewards) -> (torch.FloatTensor, List[ str ]):
    hashmap = {}
    hashes = []
    hash_rewards = torch.ones_like( rewards, dtype = torch.float32 )
    for i, response in enumerate(responses):
        images = response.images
        uid = dendrites_to_query[i]
        hashes.append([])
        # if rewards is 0 set hash_reward to 0
        if rewards[i] == 0:
            hash_rewards[i] = 0
            for j in enumerate(images):
                hashes[i].append(None)
            continue
        for j, image in enumerate(images):
            try:
                img = bt.Tensor.deserialize(image)
            except:
                bt.logging.trace(f"Detected invalid image to deserialize from dendrite {dendrites_to_query[i]}")
                hash_rewards[i] = hash_rewards[i] * 0.75
                hashes[i].append(None)
                continue
            if img.sum() == 0:
                bt.logging.trace(f"Detected black image from dendrite {dendrites_to_query[i]}")
                hash_rewards[i] = hash_rewards[i] * 0.75
                hashes[i].append(None)
                continue

            bt.logging.trace(f"Processing dendrite {uid} for image hash")
            # convert img to PIL image
            hash = imagehash.phash( transforms.ToPILImage()( img ) )
            hash = str(hash)
            if hash in hashmap:
                bt.logging.trace(f"Detected matching image from dendrite {dendrites_to_query[i]}")
                hash_rewards[i] = hash_rewards[i] * 0.75
                hash_rewards[hashmap[hash]] = hash_rewards[hashmap[hash]] * 0.75
            else:
                hashmap[hash] = i
            hashes[i].append(hash)
    return hash_rewards, hashes

def GetImageHashesOfResponses(responses):
    hashes = []
    for i, response in enumerate(responses):
        images = response.images
        hashes.append([])
        for j, image in enumerate(images):
            try:
                img = bt.Tensor.deserialize(image)
            except:
                hashes[i].append(None)
                continue
            if img.sum() == 0:
                hashes[i].append(None)
                continue

            # convert img to PIL image
            hash = imagehash.phash( transforms.ToPILImage()( img ) )
            hash = str(hash)
            hashes[i].append(hash)
    return hashes

async def forward_settings( synapse: ValidatorSettings ) -> ValidatorSettings:
    synapse._version = __version__
    synapse.nsfw_allowed = config.miner.allow_nsfw
    return synapse

def blacklist_settings( synapse: ValidatorSettings ) -> Tuple[bool, str]:
    return False, ""

def priority_settings( synapse: ValidatorSettings ) -> float:
    return 0.0

def verify_settings( synapse: ValidatorSettings ) -> None:
    pass


axon = bt.axon( config=config, wallet=wallet, ip="127.0.0.1", external_ip=bt.utils.networking.get_external_ip()).start()

# serve axon
sub.serve_axon( axon=axon, netuid=config.netuid )

while True:
     # wait for main to finish
    asyncio.run( main() )
