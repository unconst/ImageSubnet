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
import time

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

from utils import check_for_updates, __version__
bt.logging.trace(f"ImageSubnet version: {__version__}")
check_for_updates()

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
    if not config.miner.allow_nsfw or not synapse.nsfw_allowed:
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

usage_history = {} # keys will be uid, value will be list with timestamp and output size requested (in pixels)
time_to_generate_history = [] # array of tuples (timestamp, pixels, time to generate)

# whichever one of these is reached first will trigger the rate limit check of % pixels in last 5 minutes
minimum_calls_per_5_minutes = 2 # minimum number of calls before rate limit
minimum_pixels_per_5_minutes = (1024 * 1024 * 2) # minimum number of pixels before rate limit


def base_blacklist(synapse: TextToImage) -> Tuple[bool, str]:
    # check if hotkey of synapse is in meta. and if so get its position in the array
    uid = None
    axon = None
    for _uid, _axon in enumerate(meta.axons):
        if _axon.hotkey == synapse.dendrite.hotkey:
            uid = _uid
            axon = _axon
            break
    # check the stake
    tao = meta.neurons[uid].stake.tao
    if tao < config.miner.min_validator_stake:
        return True, f"stake is less than min_validator_stake ({config.miner.min_validator_stake})"
    
    # Ensure that the ip of the synapse call matches that of the axon for the validator
    if axon.ip != synapse.axon.ip:
        # if 0.0.0.0
        if axon.ip == "0.0.0.0":
            return True, "Validator has not set their ip address on the network yet, please set and try again"
        return True, f"ip of synapse call does not match ip of validator"

    return False, ""

def base_priority(synapse: TextToImage) -> float:
    uid = None
    for _uid, _axon in enumerate(meta.axons):
        if _axon.hotkey == synapse.dendrite.hotkey:
            uid = _uid
            break
    # get all neurons which have a stake higher than the min_validator_stake and be sure to include their uid position in the array (respond with tuple, of uid and neuron)
    vali_neurons = [(neuron.hotkey, neuron) for neuron in meta.neurons if neuron.stake.tao >= config.miner.min_validator_stake]

    # get total stake of all neurons
    total_stake = sum([neuron.stake.tao for neuron in meta.neurons])

    # get percentage of stake for each neuron and add to vali_neuron tuple to be (uid, neuron, percentage)
    vali_neurons = [(uid, neuron, neuron.stake.tao / total_stake) for uid, neuron in vali_neurons]
    
    # Get pixel output request of synapse
    total_pixels = synapse.height * synapse.width * synapse.num_images_per_prompt
    
    # check if the uid is in the usage_history
    if synapse.dendrite.hotkey in usage_history:
        # add the total pixels to the list of calls by that uid
        usage_history[synapse.dendrite.hotkey ].append([time.time(), total_pixels])
    # if the uid is not in the usage_history, add it
    else:
        usage_history[synapse.dendrite.hotkey] = [time.time(), total_pixels]

    # pop all calls that are older than 5 minutes
    for key in list(usage_history.keys()):
        if usage_history[key][0] < time.time() - 300:
            usage_history.pop(key)

    # check minimum calls per 5 minutes and minimum pixels per 5 minutes if both arent reached we can return false and skip the next step
    if len(usage_history) < minimum_calls_per_5_minutes:
        if sum([call[1] for calls in usage_history.values() for call in calls]) < minimum_pixels_per_5_minutes:
            return 100 + (meta.neurons[uid].stake.tao ** 0.5)


    # create new array frm vali_neurons in which all percentages of stake are set to 0 if the validator has not queried the miner yet then normalize the percentages
    normalized_vali_neurons = [(uid, neuron, percentage if uid in usage_history else 0) for uid, neuron, percentage in vali_neurons]
    # normalize the percentages
    normalized_vali_neurons = [(uid, neuron, percentage / sum([neuron[2] for neuron in normalized_vali_neurons])) for uid, neuron, percentage in normalized_vali_neurons]

    
    # get the total pixels requested by all uids in the last 5 minutes
    total_pixels_requested = sum([sum([call[1] for call in calls]) for calls in usage_history.values()])

    # get the percentage of pixels requested by each uid in the last 5 minutes
    pixels_requested_percentages = {uid: sum([call[1] for call in calls]) / total_pixels_requested for uid, calls in usage_history.items()}

    # now check if the uid is requesting more than their percentage of pixels
    if synapse.dendrite.hotkey in pixels_requested_percentages:
        if pixels_requested_percentages[synapse.dendrite.hotkey] > normalized_vali_neurons[synapse.dendrite.hotkey][2]:
            return 0
    
    # else return 100 + sqrt of stake
    return 100 + (meta.neurons[uid].stake.tao ** 0.5)


async def forward_t2i( synapse: TextToImage ) -> TextToImage:

    bt.logging.trace("Inside forward function")

    start_time = time.time()

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

    # calculate time to generate
    time_to_generate = time.time() - start_time
    # add time to generate to history but use the output_images length as the number of pixels
    total_pixels_generated = sum([image.shape[1] * image.shape[2] for image in output_images])
    time_to_generate_history.append([start_time, total_pixels_generated, time_to_generate])

    # pop all calls that are older than 15 minutes
    for call in time_to_generate_history:
        if call[0] < time.time() - 900:
            time_to_generate_history.pop(0)
    
    
    if torch.rand(1) < 0.1 and len(time_to_generate_history) > 0:
        # log out average time to generate a 512x512 image and 1024x1024 image
        time_512 = get_estimated_time_to_generate_image(512,512)
        time_1024 = get_estimated_time_to_generate_image(512,512)
        # have a random chance of logging 10% of calls
        bt.logging.trace(f"In the last 15m the average time to generate a 512x512 image took {time_512}s and a 1024x1024 image took {time_1024}s")

    return synapse

def get_estimated_time_to_generate_image(width, height):
    time_to_generate_pixel = get_time_to_generate_pixel()
    time_to_generate = time_to_generate_pixel * width * height
    return time_to_generate

def get_time_to_generate_pixel():
    time_to_generate_pixel = sum([call[2] for call in time_to_generate_history]) / sum([call[1] for call in time_to_generate_history])
    return time_to_generate_pixel

def blacklist_t2i( synapse: TextToImage ) -> Tuple[bool, str]:
    b = base_blacklist(synapse)
    if b[0]:
        return b
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
    b = base_blacklist(synapse)
    if b[0]:
        return b
    return False, ""

def priority_i2i( synapse: ImageToImage ) -> float:
    return 0.0

def verify_i2i( synapse: ImageToImage ) -> None:
    pass

async def forward_settings( synapse: MinerSettings ) -> MinerSettings:
    synapse.is_public = config.miner.public
    synapse.min_validator_stake = config.miner.min_validator_stake
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
            check_for_updates()
        sleep(1)
    # except KeyboardInterrupt:
        # continue