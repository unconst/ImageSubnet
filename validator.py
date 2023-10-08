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
from typing import List, Tuple

import asyncio
from time import sleep

bt.trace()
# Import protocol
current_script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
from protocol import TextToImage, validate_synapse, ValidatorSettings

# Load the config.
parser = argparse.ArgumentParser()
parser.add_argument( '--netuid', type = int, default = 5 )
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://finney.opentensor.ai')
parser.add_argument('--subtensor._mock', type=bool, default=False)
parser.add_argument('--validator.allow_nsfw', type=bool, default=False)
parser.add_argument('--validator.save_dir', type=str, default='./images')
parser.add_argument('--validator.save_images', type=bool, default=False)
parser.add_argument('--validator.use_absolute_size', type=bool, default=False) # Set to True if you want to 100% match the input size, else just match the aspect ratio
parser.add_argument('--validator.label_images', type=bool, default=False, help="if true, label images with dendrite uid and score")
parser.add_argument('--device', type=str, default='cuda')
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
meta.sync()
dend = bt.dendrite( wallet = wallet )

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from utils import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from fabric.utils import get_free_gpu, tile_images
import matplotlib.font_manager as fm


DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")

def get_system_fonts():
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    return font_list

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


# Init the validator weights.
alpha = 0.01
# weights = torch.rand_like( meta.uids, dtype = torch.float32 )
weights = torch.ones_like( meta.uids , dtype = torch.float32 ) * 0.5

# multiply weights by the active tensor
curr_block = sub.block

# loop over all last_update, any that are within 600 blocks are set to 1 others are set to 0 
weights = weights * meta.last_update > curr_block - 600

# all nodes with more than 1e3 total stake are set to 0 (sets validtors weights to 0)
weights = weights * meta.total_stake < 1.024e3

 # Amount of images
num_images = 1
dendrites_per_query = 25

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
    queryable_uids = meta.last_update > curr_block - 600 * meta.total_stake < 1.024e3

    # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
    zipped_uids = list(zip(uids, queryable_uids))
    filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
    dendrites_to_query = random.sample( uids, min( dendrites_per_query, len(filtered_uids) ) )

    # Generate a random synthetic prompt. cut to first 20 characters.
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
        height = width,
        width = height,
        negative_prompt = "",
        nsfw_allowed=config.validator.allow_nsfw,
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
        timeout = timeout
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
    dissimilarity_rewards: torch.FloatTensor = calculate_dissimilarity_rewards( best_images )

    # Calculate the final rewards.
    dissimilarity_weight = 0.15
    rewards = (1 - dissimilarity_weight) * rewards + dissimilarity_weight * dissimilarity_rewards    
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

async def forward_settings( synapse: ValidatorSettings ) -> ValidatorSettings:
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
