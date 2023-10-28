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
from protocol import TextToImage, validate_synapse, ValidatorSettings
from utils import get_device, get_scoring_model, check_for_updates, __version__, total_dendrites_per_query, minimum_dendrites_per_query, num_images, calculate_rewards_for_prompt_alignment, calculate_dissimilarity_rewards, get_system_fonts
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


import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from utils import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from fabric.utils import get_free_gpu, tile_images
import matplotlib.font_manager as fm

DEVICE = get_device(config)

# For image to text generation.
# Load the scoring model
scoring_model = get_scoring_model(config)

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
sizes = [512, 768, 1024, 1280]

# list of aspect ratios [(1, 1), (4, 3), (16, 9)]
aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]

def get_resolution(size_index = None, aspect_ratio_index = None):
    # pick a random size and aspect ratio
    if size_index is None or size_index >= len(sizes):
        size_index = random.randint(0, len(sizes)-1)
    if size_index == 0 and aspect_ratio_index is None:
        aspect_ratio_index = 0
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


# Init the validator weights.
alpha = 0.01
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

last_updated_block = curr_block - (curr_block % 100)
last_reset_weights_block = curr_block


def add_black_border(image, border_size):
    # Create a new image with the desired dimensions
    new_width = image.width
    new_height = image.height + border_size
    new_image = Image.new("RGB", (new_width, new_height), color="black")
    
    # Paste the original image onto the new image, leaving space at the top for the border
    new_image.paste(image, (0, border_size))
    
    return new_image


safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( DEVICE )
processor = CLIPImageProcessor()

# find DejaVu Sans font
if (config.validator.label_images == True):
    fonts = get_system_fonts()
    dejavu_font = None
    for font in fonts:
        if "DejaVu" in font:
            dejavu_font = font
            break 

    default_font = ImageFont.truetype(dejavu_font, 30)

async def main():
    global weights, last_updated_block, last_reset_weights_block
    # every 10 blocks, sync the metagraph.
    if sub.block % 10 == 0:
        # create old list of (uids, hotkey)
        old_uids = list(zip(meta.uids.tolist(), meta.hotkeys))
        meta.sync(subtensor = sub, )
        # create new list of (uids, hotkey)
        new_uids = list(zip(meta.uids.tolist(), meta.hotkeys))
        # if the lists are different, reset weights for that uid
        for i in range(len(old_uids)):
            if old_uids[i] != new_uids[i]:
                weights[i] = 0.3 * torch.median( weights[weights != 0] )

    uids = meta.uids.tolist() 

    # if uids is longer than weight matrix, then we need to add more weights.
    if len(uids) > len(weights):
        bt.logging.trace("Adding more weights")
        size_difference = len(uids) - len(weights)
        new_weights = torch.zeros( size_difference, dtype = torch.float32 )
        # the new weights should be 0.3 * the median of all non 0 weights
        new_weights = new_weights + 0.3 * torch.median( weights[weights != 0] )
        weights = torch.cat( (weights, new_weights) )
        del new_weights

    bt.logging.trace('uids')
    bt.logging.trace(uids)

    # Select up to dendrites_per_query random dendrites.
    queryable_uids = (meta.last_update > curr_block - 600) * (meta.total_stake < 1.024e3)

    # for all uids, check meta.neurons[uid].axon_info.ip == '0.0.0.0' if so, set queryable_uids[uid] to false
    queryable_uids = queryable_uids * torch.Tensor([meta.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])

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

    timeout_increase = 1

    if dendrites_per_query > active_miners:
        bt.logging.warning(f"Warning: not enough active miners to sufficently validate images, rewards may be inaccurate. Active miners: {active_miners}, Minimum per query: {minimum_dendrites_per_query}")
    elif active_miners < dendrites_per_query * 3:
        bt.logging.warning(f"Warning: not enough active miners, miners may be overloaded from other validators. Enabling increased timeout.")
        timeout_increase = 2

    # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
    zipped_uids = list(zip(uids, queryable_uids))
    filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
    dendrites_to_query = random.sample( filtered_uids, min( dendrites_per_query, len(filtered_uids) ) )

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

    bt.logging.trace(f"Inital prompt: {initial_prompt}\nPrompt: {prompt}\n")

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

    # total pixels
    total_pixels = query.height * query.width

    base_timeout = 12
    base_timeout_size = 512*512

    max_timeout = 30

    # calculate timeout based on size of image, image size goes up quadraticly but timeout goes up linearly, so if you go from 512,512 -> 1024,1024, the timeout should be 3x
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

    bt.logging.trace("Calling dendrite pool")
    bt.logging.trace(f"Query: {query.text}")
    bt.logging.trace("Dendrites:")
    bt.logging.trace(dendrites_to_query)

    # Get response from endpoint.
    responses = await dendrite_pool.async_forward(
        uids = dendrites_to_query,
        query = query,
        timeout = timeout * timeout_increase
    )

    # validate all responses, if they fail validation remove both the response from responses and dendrites_to_query
    for i, response in enumerate(responses):
        valid, error = validate_synapse(response)
        if not valid:
            bt.logging.trace(f"Detected invalid response from dendrite {dendrites_to_query[i]}: {error}")
            del responses[i]
            del dendrites_to_query[i]

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

    (rewards, best_images) = calculate_rewards_for_prompt_alignment( query, responses )
    rewards = rewards / torch.max(rewards)

    # zip rewards and images together, then filter out all images which have a reward of 0
    zipped_rewards = list(zip(rewards, best_images))
    filtered_rewards = list(zip(*filter(lambda x: x[0] != 0, zipped_rewards)))
    # get back images
    filtered_best_images = filtered_rewards[1]

    dissimilarity_rewards: torch.FloatTensor = calculate_dissimilarity_rewards( filtered_best_images )

    # dissimilarity isnt the same length because we filtered out images with 0 reward, so we need to create a new tensor of length rewards
    new_dissimilarity_rewards = torch.zeros( len(rewards), dtype = torch.float32 )
    y = 0
    for i, reward in enumerate(rewards):
        if reward != 0:
            new_dissimilarity_rewards[i] = dissimilarity_rewards[y]
            y+=1

    dissimilarity_rewards = new_dissimilarity_rewards

    dissimilarity_rewards = dissimilarity_rewards / torch.max(dissimilarity_rewards)

    # my goal with dissimilarity_rewards is to encourage diversity in the images

    # normalize rewards such that the highest value is 1

    dissimilarity_weight = 0.15
    rewards = rewards + dissimilarity_weight * dissimilarity_rewards

    # Perform imagehash (perceptual hash) on all images. Any matching images are given a reward of 0.
    hash_rewards, _ = ImageHashRewards(dendrites_to_query, responses, rewards)

    # multiply rewards by hash rewards
    rewards = rewards * hash_rewards

    rewards = rewards / torch.max(rewards)
    bt.logging.trace("Rewards:")
    bt.logging.trace(rewards)
    
    if torch.sum( rewards ) == 0:
        bt.logging.trace("All rewards are 0, skipping block")
        return
    

    # reorder rewards to match dendrites_to_query
    _rewards = torch.zeros( len(uids), dtype = torch.float32 )
    for i, uid in enumerate(dendrites_to_query):
        _rewards[uids.index(uid)] = rewards[i]
    rewards = _rewards
    
    weights = weights + alpha * rewards

    # loop through all images and remove black images
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

    # if save images is true, save the images to a folder
    if config.validator.save_images == True:
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


    # every loop scale weights by 0.993094, sets half life to 100 blocks
    weights = weights * 0.993094

    # Optionally set weights
    current_block = sub.block
    if current_block - last_updated_block  >= 100:
        bt.logging.trace(f"Setting weights")

        # Normalize weights.
        weights = weights / torch.sum( weights )

        # TODO POTENTIALLY ADD THIS IN LATER
        # any weights higher than (1 / len(weights)) * 10 are set to (1 / len(weights)) * 10
        # scale_max = (1 / len(weights)) * (len(weights) * 0.0390625)
        # weights[weights > scale_max] = scale_max 

        # # normalize again
        # weights = weights / torch.sum( weights )

        bt.logging.trace("Weights:")
        bt.logging.trace(weights)

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
        last_updated_block = current_block
        check_for_updates()

def ImageHashRewards(dendrites_to_query, responses, rewards) -> (torch.FloatTensor, List[ str ]):
    hashmap = {}
    hashes = []
    hash_rewards = torch.ones_like( rewards, dtype = torch.float32 )
    for i, response in enumerate(responses):
        images = response.images
        uid = dendrites_to_query[i]
        hashes.append([])
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
            hash = imagehash.phash( transforms.ToPILImage()( img ) )
            if hash in hashmap:
                bt.logging.trace(f"Detected matching image from dendrite {dendrites_to_query[i]}")
                hash_rewards[i] = hash_rewards[i] * 0.75
                hash_rewards[hashmap[hash]] = hash_rewards[hashmap[hash]] * 0.75
            else:
                hashmap[hash] = i
            hashes[i].append(hash)
    return hash_rewards, hashes

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
