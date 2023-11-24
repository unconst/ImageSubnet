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

from protocol import TextToImage, validate_synapse, ValidatorSettings, ImageToImage

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
    queryable_uids = [12]

    for prompt in prompts:
        query = create_query(prompt)
        curr_block = subtensor.block

        bt.logging.trace("Sending out request... 30s timeout")
        responses = await dendrite_pool.async_forward( query=query, uids=queryable_uids, timeout=30 )
        bt.logging.trace("Responses finished!")

        # get response of each dendrite and do imagetoimage
        for uid in responses:
            response = responses[uid]
            if response is None:
                bt.logging.warning(f"Got no response from {uid}")
                continue
            if not validate_synapse(response):
                bt.logging.warning(f"Got invalid response from {uid}")
                continue
            images = response.images
            image = images[0]
            # image is of type bittensor.Tensor
            print(type(image), "type of response image")
            # send imagetoimage
            query = ImageToImage(
                image = image,
                text = prompt,
                negative_prompt = "",
                height = 768,
                width = 768,
                num_images_per_prompt = 1,
                seed = -1,
                )

            print(query)

            responses = await dendrite_pool.async_forward( query=query, uids=[uid], timeout=30 )

            for uid in responses:
                if response is None:
                    bt.logging.warning(f"Got no response from {uid} i2i")
                    continue
                if not validate_synapse(response):
                    bt.logging.warning(f"Got invalid response from {uid} i2i")
                    continue
                images = response.images
                image = images[0]
                # image is of type bittensor.Tensor
                print(type(image), "type of response image i2i")

print("Running main")
import asyncio

# loop through all prompts
asyncio.run(main())