import bittensor as bt
import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image

# Get the current script's directory (assuming miner.py is in the miners folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the project's root directory to sys.path
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torchvision.transforms as transforms
from dendrite import AsyncDendritePool
from typing import List, Tuple

from protocol import TextToImage, validate_synapse, ValidatorSettings

from utils import calculate_rewards_for_prompt_alignment, calculate_dissimilarity_rewards
from fabric.utils import tile_images


bt.trace()

wallet = bt.wallet(  )
metagraph = bt.metagraph( 5, network="finney" )
subtensor = bt.subtensor( network = "finney" )

metagraph.sync(subtensor=subtensor)
bt.logging.trace("Synced!")

prompts = ["anime girl, anime style drawing, cherry blossom in the background"]

# check if ./prompts.txt exists, if so, read it and set prompts to that
if os.path.exists('./prompts.txt'):
    bt.logging.trace("Found prompts.txt, using prompts from there")
    with open('./prompts.txt') as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
else:
    bt.logging.trace("No prompts.txt found, using default prompts")

uids = metagraph.uids

def create_query(prompt):
    query = TextToImage(
        text = prompt,
        negative_prompt = "",
        height = 768,
        width = 768,
        num_images_per_prompt = 1,
        seed = -1,
        )
    return query

selected_uids = None
# load uids.txt, these are the uids of the dendrites we want to query
if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "uids.txt")):
    bt.logging.trace("Found uids.txt, using uids from there")
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "uids.txt")) as f:
        selected_uids = f.readlines()
        selected_uids = [uid.strip() for uid in selected_uids]
        # all uids should be ints, if cant convert to int, remove from list
        selected_uids = [int(uid) for uid in selected_uids if uid.isdigit()]
    if len(selected_uids) == 0:
        selected_uids = None
        bt.logging.trace("No valid uids found in uids.txt, using all uids")

dendrite_pool = AsyncDendritePool( wallet = wallet, metagraph = metagraph)

async def main():
    curr_block = subtensor.block
    # Init the validator weights.
    alpha = 0.01
    # weights = torch.rand_like( meta.uids, dtype = torch.float32 )
    weights = torch.ones_like( uids , dtype = torch.float32 )

    # set all nodes without ips set to 0
    weights = weights * torch.Tensor([metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in metagraph.uids]) * 0.5


    # Select up to dendrites_per_query random dendrites.
    queryable_uids = (metagraph.last_update > curr_block - 600) * (metagraph.total_stake < 1.024e3)

    # for all uids, check meta.neurons[uid].axon_info.ip == '0.0.0.0' if so, set queryable_uids[uid] to false
    queryable_uids = queryable_uids[:len(metagraph.neurons)] * torch.Tensor([metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids][:len(queryable_uids)])

    # set all uids not in selected_uids to 0
    if selected_uids is not None:
        for uid in uids:
            if uid not in selected_uids:
                queryable_uids[list(uids).index(uid)] = 0

    emissions = metagraph.emission
            
    zipped_uids = list(zip(metagraph.uids, queryable_uids))
    filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]

    # get emissions for all filtered uids
    filtered_emissions = [emissions[list(uids).index(uid)] for uid in filtered_uids]

    queryable_uids = filtered_uids

    for prompt in prompts:
        query = create_query(prompt)
        curr_block = subtensor.block

        bt.logging.trace("Sending out request... 30s timeout")
        responses = await dendrite_pool.async_forward( query=query, uids=queryable_uids, timeout=30 )
        bt.logging.trace("Responses finished!")
        # create sub folder called "imgoutputs" put all images in there

        for i, response in enumerate(responses):
            valid, error = validate_synapse(response)
            if not valid:
                bt.logging.trace(f"Detected invalid response from dendrite {queryable_uids[i]}: {error}")
                del responses[i]
        bt.logging.trace("Calculating rewards")
        (rewards, best_images) = calculate_rewards_for_prompt_alignment( query, responses )
        rewards = rewards / torch.max(rewards)
        old_rewards = rewards
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

        inverted_rewards = old_rewards + dissimilarity_weight * dissimilarity_rewards * -1

        rewards = rewards / torch.max(rewards)

        inverted_rewards = inverted_rewards / torch.max(inverted_rewards)

        bt.logging.trace("Rewards:")
        bt.logging.trace(rewards)
        
        if torch.sum( rewards ) == 0:
            bt.logging.trace("All rewards are 0, skipping block")
            return


        _uids = list(queryable_uids)
        # reorder rewards to match dendrites_to_query
        _rewards = torch.zeros( len(uids), dtype = torch.float32 )
        for i, uid in enumerate(queryable_uids):
            _rewards[_uids.index(uid)] = rewards[i]
        rewards = _rewards
        
        weights = weights + alpha * rewards

        # Normalize the weights.
        weights = weights / torch.sum( weights )

        # create imgoutputs if not exists
        if not os.path.exists('imgoutputs'):
            os.makedirs('imgoutputs')
        if not os.path.exists(f'imgoutputs/{curr_block}'):
            os.makedirs(f'imgoutputs/{curr_block}')

        # pad to 4 digits
        def pad(uid: int) -> str:
            return str(uid.item() if isinstance(uid, torch.Tensor) else uid).zfill(4) 

        def trim(reward: float) -> str:
            return str(reward.item() if isinstance(reward, torch.Tensor) else reward)[:5]

        # save all images by `imgoutputs/0[uid]-[score].png`
        for i, image in enumerate(best_images):
            if image is None:
                continue
            try:
                # resize image to 512x512
                image = image.resize((512, 512))
                image.save(f'imgoutputs/{curr_block}/{pad(_uids[i])}-{trim(old_rewards[i])}-{trim(rewards[i])}-{trim(inverted_rewards[i])}.png')
                bt.logging.trace(f"Saved image for uid {_uids[i]}")
            except:
                continue

        # zip best_images with _uids then sort by filtered_emissions of the uid
        zipped = list(zip(best_images, _uids))
        zipped.sort(key=lambda x: filtered_emissions[filtered_uids.index(x[1])], reverse=True)

        # get back images and tile them
        images = list(zip(*zipped))[0]
        # resize to 0.5x0.5
        # images = [image.resize((int(image.width * 0.5), int(image.height * 0.5))) for image in images]
        # the above code doesnt take into accound that the image could be None, replace all None with blank black images of size width * 0.5, height * 0.5
        for i, image in enumerate(images):
            if image is None:
                images[i] = Image.new('RGB', (int(query.width * 0.5), int(query.height * 0.5)), (0, 0, 0))
            else:
                images[i] = image.resize(int(query.width * 0.5), int(query.height * 0.5))
        # tile images
        tiled_image = tile_images(images)
        # save tiled image
        tiled_image.save(f'imgoutputs/{curr_block}/tiled.png')

        # save a file with all uids and scores
        with open(f'imgoutputs/{curr_block}/scores.txt', 'w') as f:
            for i, uid in enumerate(_uids):
                f.write(f'{uid},{trim(old_rewards[i])},{trim(rewards[i])},{trim(inverted_rewards[i])}\n')

        # write prompt to file
        with open(f'imgoutputs/{curr_block}/prompt.txt', 'w') as f:
            f.write(prompt)
        bt.logging.trace("Finished!")

print("Running main")
import asyncio

# loop through all prompts
asyncio.run(main())