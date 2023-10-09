# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Confirm minimum python version
import sys
import os 

# Get the current script's directory (assuming miner.py is in the miners folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the project's root directory to sys.path
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

required_version = (3, 9)

if sys.version_info < required_version:
    raise Exception(f"Python version {required_version[0]}.{required_version[1]} or higher is required.")

# Imports
import torch
import bittensor as bt
from time import sleep
from typing import Tuple, Union

bt.debug()
bt.trace()

from config import config

from utils import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from protocol import TextToImage, ImageToImage, validate_synapse, MinerSettings

from generate import t2i, i2i

subtensor = bt.subtensor( config.subtensor.chain_endpoint, config=config )
meta = subtensor.metagraph( config.netuid )
wallet = bt.wallet( config=config )

transform = transforms.Compose([
    transforms.PILToTensor()
])

DEVICE = torch.device(config.device)

bt.logging.trace("Loading safety checker")
safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( DEVICE )
processor = CLIPImageProcessor()

last_updated_block = subtensor.block - 100

if config.miner.allow_nsfw:
    bt.logging.warning("NSFW is enabled. Without a filter, your miner may generate unwanted images. Please use with caution.")


def CheckNSFW(output, synapse):
    if not config.miner.allow_nsfw or not synapse.allow_nsfw:
        print(output.images, "output images")
        clip_input = processor([transform(image) for image in output.images], return_tensors="pt").to( DEVICE )
        images, has_nsfw_concept = safetychecker.forward( images=output.images, clip_input=clip_input.pixel_values.to( DEVICE ))
        return has_nsfw_concept
    else:
        return [False] * len(output.images)

class Images():
    def __init__(self, images):
        self.images = images

def GenerateImage(synapse, generator):
    height = synapse.height
    bt.logging.trace("Inside GenerateImage")
    if config.miner.height.min is not None and height < config.miner.height.min:
        if synapse.fixed_resolution:
            raise ValueError(f"height ({height}) must be greater than or equal to height.min ({config.miner.height.min})")
        else:
            height = config.miner.height.min
    elif config.miner.height.max is not None and height > config.miner.height.max:
        raise ValueError(f"height ({height}) must be less than or equal to height.max ({config.miner.height.max})")
    
    width = synapse.width
    if config.miner.width.min is not None and width < config.miner.width.min:
        if synapse.fixed_resolution:
            raise ValueError(f"width ({width}) must be greater than or equal to width.min ({config.miner.width.min})")
        else:
            width = config.miner.width.min
    elif config.miner.width.max is not None and width > config.miner.width.max:
        raise ValueError(f"width ({width}) must be less than or equal to width.max ({config.miner.width.max})")
    
    # if height and widht are different from synapse, ensure that the aspect ratio is the same to the nearest divisible by 8
    if height != synapse.height or width != synapse.width:
        # determine the aspect ratio of the synapse
        aspect_ratio = synapse.width / synapse.height
        # determine the aspect ratio of the new height and width
        new_aspect_ratio = width / height
        # if the aspect ratio is different, we need to adjust the height or width to match the aspect ratio of the synapse
        if aspect_ratio != new_aspect_ratio:
            # if the new aspect ratio is greater than the synapse aspect ratio, we need to reduce the width
            if new_aspect_ratio > aspect_ratio:
                # reduce the width to the nearest divisible by 8
                width = int(width - (width % 8))
                # calculate the new height
                height = int(width / aspect_ratio)
            # if the new aspect ratio is less than the synapse aspect ratio, we need to reduce the height
            else:
                # reduce the height to the nearest divisible by 8
                height = int(height - (height % 8))
                # calculate the new width
                width = int(height * aspect_ratio)

    num_images_per_prompt = synapse.num_images_per_prompt
    if config.miner.max_images is not None and num_images_per_prompt > config.miner.max_images:
        print(f"num_images_per_prompt ({num_images_per_prompt}) must be less than or equal to max_images ({config.miner.max_images}), reducing num_images_per_prompt")
        num_images_per_prompt = config.miner.max_images

    # determine total pixels to generate
    total_pixels = height * width * synapse.num_images_per_prompt
    if config.miner.max_pixels is not None and total_pixels > config.miner.max_pixels:
        raise ValueError(f"total pixels ({total_pixels}) must be less than or equal to max_pixels ({config.miner.max_pixels}), reduce image size, or num_images_per_prompt")
    
    bt.logging.trace("Attempting to generate image")
    try:
        # If we are doing image to image, we need to use a different pipeline.
        image = synapse.image
        output_images = i2i(synapse)
    except AttributeError as e:
        # run normal text to image pipeline
        output_images = t2i(synapse)

    bt.logging.trace("Image generated")
    print(output_images)
        
   
    
    return Images(output_images)

async def forward_t2i( synapse: TextToImage ) -> TextToImage:

    bt.logging.trace("Inside forward function")

    seed = synapse.seed

    # Let's set a seed for reproducibility.
    if(seed == -1):
        seed = torch.randint(1000000000, (1,)).item()

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    output = GenerateImage(synapse, generator)
    print(output.images)

    has_nsfw_concept = CheckNSFW(output, synapse) # will return all False if allow_nsfw is enabled
    if any(has_nsfw_concept):
        output.images = [image for image, has_nsfw in zip(output.images, has_nsfw_concept) if not has_nsfw]
        # try to regenerate another image once
        output2 = GenerateImage(synapse, generator)
        has_nsfw_concept = CheckNSFW(output2, synapse)
        if any(has_nsfw_concept):
            output2.images = [image for image, has_nsfw in zip(output2.images, has_nsfw_concept) if not has_nsfw]
            output.images += output2.images
        if len(output.images) == 0:
            # if we still have no images, just return the original output
            bt.logging.warning("All images were NSFW, returning empty list")
            output.images = []

    # copy output images to new array
    output_images = output.images.copy()
    # clear synapse images
    synapse.images = []

    for image in output_images:
        img_tensor = transform(image)
        synapse.images.append( bt.Tensor.serialize( img_tensor ) )

    # validate the synapse
    valid, error = validate_synapse(synapse)
    if not valid:
        raise ValueError(f"Invalid synapse: {error}")

    return synapse

def blacklist_t2i( synapse: TextToImage ) -> Tuple[bool, str]:
    return False, ""

def priority_t2i( synapse: TextToImage ) -> float:
    return 0.0

def verify_t2i( synapse: TextToImage ) -> None:
    pass


async def forward_i2i( synapse: ImageToImage ) -> ImageToImage:

    bt.logging.trace("Inside forward function")

    seed = synapse.seed

    # Let's set a seed for reproducibility.
    if(seed == -1):
        seed = torch.randint(1000000000, (1,)).item()

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    output = GenerateImage(synapse, generator)
    print(output.images)

    has_nsfw_concept = CheckNSFW(output, synapse) # will return all False if allow_nsfw is enabled
    if any(has_nsfw_concept):
        output.images = [image for image, has_nsfw in zip(output.images, has_nsfw_concept) if not has_nsfw]
        # try to regenerate another image once
        output2 = GenerateImage(synapse, generator)
        has_nsfw_concept = CheckNSFW(output2, synapse)
        if any(has_nsfw_concept):
            output2.images = [image for image, has_nsfw in zip(output2.images, has_nsfw_concept) if not has_nsfw]
            output.images += output2.images
        if len(output.images) == 0:
            # if we still have no images, just return the original output
            bt.logging.warning("All images were NSFW, returning empty list")
            output.images = []

    # copy output images to new array
    output_images = output.images.copy()
    # clear synapse images
    synapse.images = []

    for image in output_images:
        img_tensor = transform(image)
        synapse.images.append( bt.Tensor.serialize( img_tensor ) )

    return synapse


def blacklist_i2i( synapse: ImageToImage ) -> Tuple[bool, str]:
    return False, ""

def priority_i2i( synapse: ImageToImage ) -> float:
    return 0.0

def verify_i2i( synapse: ImageToImage ) -> None:
    pass

async def forward_settings( synapse: MinerSettings ) -> MinerSettings:
    synapse.nsfw_allowed = config.miner.allow_nsfw
    synapse.max_images = config.miner.max_images
    synapse.max_pixels = config.miner.max_pixels
    synapse.min_width = config.miner.width.min
    synapse.max_width = config.miner.width.max
    synapse.min_height = config.miner.height.min
    synapse.max_height = config.miner.height.max
    return synapse

def blacklist_settings( synapse: MinerSettings ) -> Tuple[bool, str]:
    return False, ""

def priority_settings( synapse: MinerSettings ) -> float:
    return 0.0

def verify_settings( synapse: MinerSettings ) -> None:
    pass


wallet = bt.wallet( config=config )
axon = bt.axon( config=config, wallet=wallet, ip="127.0.0.1", external_ip=bt.utils.networking.get_external_ip()).attach( forward_t2i, blacklist_t2i, priority_t2i, verify_t2i ).attach( forward_i2i, blacklist_i2i, priority_i2i, verify_i2i ).attach( forward_settings, blacklist_settings, priority_settings, verify_settings ).start()

# serve axon
subtensor.serve_axon( axon=axon, netuid=config.netuid )

# keep process alive
bt.logging.trace('Miner running. ^C to exit.')

while True:
    # try:
        if subtensor.block - last_updated_block >= 100:
            bt.logging.trace(f"Setting miner weight")
            # find the uid that matches config.wallet.hotkey [meta.axons[N].hotkey == config.wallet.hotkey]
            # set the weight of that uid to 1.0
            uid = None
            try:
                for _uid, axon in enumerate(meta.axons):
                    if axon.hotkey == wallet.hotkey.ss58_address:
                        # uid = axon.uid
                        # uid doesnt exist ona xon
                        uid = _uid
                        break
                if uid is not None:
                    # 0 weights for all uids
                    weights = torch.Tensor([0.0] * len(meta.uids))
                    # 1 weight for uid
                    weights[uid] = 1.0
                    (uids, processed_weights) = bt.utils.weight_utils.process_weights_for_netuid( uids = meta.uids, weights = weights, netuid=config.netuid, subtensor = subtensor)
                    subtensor.set_weights(wallet = wallet, netuid = config.netuid, weights = processed_weights, uids = uids)
                    last_updated_block = subtensor.block
                    bt.logging.trace("Miner weight set!")
                else:
                    bt.logging.warning(f"Could not find uid with hotkey {config.wallet.hotkey} to set weight")
            except Exception as e:
                bt.logging.warning(f"Could not set miner weight: {e}")
                raise e
                # pass
        sleep(1)
    # except KeyboardInterrupt:
        # continue