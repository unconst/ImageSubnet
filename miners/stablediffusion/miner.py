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
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


required_version = (3, 9)

if sys.version_info < required_version:
    raise Exception(f"Python version {required_version[0]}.{required_version[1]} or higher is required.")

# Imports
import torch
import bittensor as bt
import argparse
from time import sleep
from typing import Tuple

bt.debug()
bt.trace()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--miner.allow_nsfw', type=bool, default=False)
parser.add_argument('--miner.max_batch_size', type=int, default=1)
parser.add_argument('--miner.model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
parser.add_argument('--miner.model_type', type=str, default='XL') # XL, 1.5, 2.0
parser.add_argument('--miner.vae', type=str, default=None)
parser.add_argument('--miner.steps.max', type=int, default=50)
parser.add_argument('--miner.steps.min', type=int, default=10)
parser.add_argument('--miner.guidance.max', type=float, default=None)
parser.add_argument('--miner.guidance.min', type=float, default=0)
parser.add_argument('--miner.height.max', type=int, default=2048)
parser.add_argument('--miner.height.min', type=int, default=None)
parser.add_argument('--miner.width.max', type=int, default=2048)
parser.add_argument('--miner.width.min', type=int, default=None)
parser.add_argument('--miner.max_images', type=int, default=4)
parser.add_argument('--miner.inference_steps', type=int, default=20)
parser.add_argument('--miner.guidance_scale', type=int, default=7.5)
parser.add_argument('--miner.max_pixels', type=int, default=(1024 * 1024 * 4)) # determines total number of images able to generate in one batch (height * width * num_images_per_prompt)
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://entrypoint-finney.opentensor.ai:443')
parser.add_argument('--wallet.hotkey', type=str, default='default')
parser.add_argument('--wallet.name', type=str, default='default')
parser.add_argument('--wallet.path', type=str, default='~/.bittensor/wallets')
parser.add_argument('--netuid', type=int, default=5)
parser.add_argument('--axon.port', type=int, default=3000)

config = bt.config( parser )
subtensor = bt.subtensor( config.subtensor.chain_endpoint, config=config )
meta = subtensor.metagraph( config.netuid )
wallet = bt.wallet( config=config )


# if model_type is not ['XL', '1.5', or '2.0'], then we will error and provide the values that are allowed
if config.miner.model_type not in ['XL', '1.5', '2.0']:
    raise argparse.ArgumentTypeError(f"model_type must be XL, 1.5, or 2.0, but got {config.miner.model_type}")

# verify min/max height and width as they should all be divisible by 8
if config.miner.height.max is not None and config.miner.height.max % 8 != 0:
    raise argparse.ArgumentTypeError(f"height.max must be divisible by 8, but got {config.miner.height.max}")
if config.miner.height.min is not None and config.miner.height.min % 8 != 0:
    raise argparse.ArgumentTypeError(f"height.min must be divisible by 8, but got {config.miner.height.min}")
if config.miner.width.max is not None and config.miner.width.max % 8 != 0:
    raise argparse.ArgumentTypeError(f"width.max must be divisible by 8, but got {config.miner.width.max}")
if config.miner.width.min is not None and config.miner.width.min % 8 != 0:
    raise argparse.ArgumentTypeError(f"width.min must be divisible by 8, but got {config.miner.width.min}")


from utils import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

DEVICE = torch.device(config.device)

bt.logging.trace("Loading safety checker")
safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( DEVICE )
processor = CLIPImageProcessor()

if config.miner.allow_nsfw:
    bt.logging.warning("NSFW is enabled. Without a filter, your miner may generate unwanted images. Please use with caution.")

# Stable diffusion
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLPipeline, AutoencoderKL

import torchvision.transforms as transforms
from protocol import TextToImage

bt.logging.trace("Loading model: {}".format(config.miner.model))

model_path = config.miner.model
# Lets instantiate the stable diffusion model.
if model_path.endswith('.safetensors') or model_path.endswith('.ckpt'):
    # Load from local file or from url.
    if config.miner.model_type == 'XL':
        model = StableDiffusionXLPipeline.from_single_file( model_path ).to( DEVICE )
    else:
        model = StableDiffusionPipeline.from_ckpt( model_path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False ).to( DEVICE )
else:
    # Load from huggingface model hub.
    model =  StableDiffusionPipeline.from_pretrained( model_path , custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False ).to( DEVICE )

if config.miner.vae is not None:
    model.vae = AutoencoderKL.from_single_file( config.miner.vae ).to( DEVICE )

if config.miner.model_type == 'XL':
    img2img = StableDiffusionXLPipeline(**model.components)
else:
    img2img = StableDiffusionImg2ImgPipeline(**model.components)


transform = transforms.Compose([
    transforms.PILToTensor()
])

async def f( synapse: TextToImage ) -> TextToImage:

    seed = synapse.seed

    # Let's set a seed for reproducibility.
    if(seed == -1):
        seed = torch.randint(1000000000, (1,)).item()

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Check if the batch size is valid.
    if synapse.num_images_per_prompt > config.miner.max_batch_size:
        raise ValueError(f"num_images_per_prompt ({synapse.num_images_per_prompt}) must be less than or equal to max_batch_size ({config.miner.max_batch_size})")

    output = GenerateImage(synapse, generator)

    synapse.images = []

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



    for image in output.images:
        img_tensor = transform(image)
        synapse.images.append( bt.Tensor.serialize( img_tensor ) )

    return synapse

def CheckNSFW(output, synapse):
    if not config.miner.allow_nsfw or not synapse.allow_nsfw:
        clip_input = processor([transform(image) for image in output.images], return_tensors="pt").to( DEVICE )
        images, has_nsfw_concept = safetychecker.forward( images=output.images, clip_input=clip_input.pixel_values.to( DEVICE ))
        return has_nsfw_concept
    else:
        return [False] * len(output.images)

def GenerateImage(synapse, generator):
    inference_steps = synapse.num_inference_steps
    if config.miner.steps.min is not None and inference_steps < config.miner.steps.min:
        inference_steps = config.miner.steps.min
    elif config.miner.steps.max is not None and inference_steps > config.miner.steps.max:
        inference_steps = config.miner.steps.max

    guidance_scale = synapse.guidance_scale
    if config.miner.guidance.min is not None and guidance_scale < config.miner.guidance.min:
        guidance_scale = config.miner.guidance.min
    elif config.miner.guidance.max is not None and guidance_scale > config.miner.guidance.max:
        guidance_scale = config.miner.guidance.max

    height = synapse.height
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
    
    try:
        # If we are doing image to image, we need to use a different pipeline.
        output = img2img(
            image = synapse.image,
            num_images_per_prompt = num_images_per_prompt,
            num_inference_steps = inference_steps,
            guidance_scale = guidance_scale,
            strength = synapse.strength,
            negative_prompt = synapse.negative_prompt,
            generator = generator
        )
    except:
        output = model(
            prompt = synapse.text,
            height = height,
            width = width,
            num_images_per_prompt = num_images_per_prompt,
            num_inference_steps = inference_steps,
            guidance_scale = guidance_scale,
            negative_prompt = synapse.negative_prompt,
            generator = generator
        )
        
    synapse.images = []
    for image in output.images:
        img_tensor = transform(image)
        synapse.images.append( bt.Tensor.serialize( img_tensor ) )

    return synapse

def b( synapse: TextToImage ) -> Tuple[bool, str]:
    return False, ""

def p( synapse: TextToImage ) -> float:
    return 0.0

def v( synapse: TextToImage ) -> None:
    pass

wallet = bt.wallet( config=config )
axon = bt.axon( config=config, wallet=wallet, ip=bt.utils.networking.get_external_ip()).attach( f, b, p, v ).start()

# serve axon
subtensor.serve_axon( axon=axon, netuid=config.netuid )

# keep process alive
bt.logging.trace('Miner running. ^C to exit.')

while True:
    try:
        if meta.block % 100 == 0:
            bt.logging.trace(f"Setting miner weight")
            # find the uid that matches config.wallet.hotkey [meta.axons[N].hotkey == config.wallet.hotkey]
            # set the weight of that uid to 1.0
            uid = None
            for axon in meta.axons:
                if axon.hotkey == wallet.hotkey:
                    uid = axon.uid
                    break
            if uid is not None:
                # 0 weights for all uids
                weights = torch.Tensor([0.0] * len(meta.uids))
                # 1 weight for uid
                weights[uid] = 1.0
                processed_weights = bt.utils.weight_utils.process_weights_for_netuid( uids = meta.uids, weights = weights, netuid=config.netuid, subtensor = subtensor)
                meta.set_weights(wallet = wallet, netuid = config.netuid, weights = processed_weights, uids = meta.uids)
                bt.logging.trace("Miner weight set!")
            else:
                bt.logging.warning(f"Could not find uid with hotkey {config.wallet.hotkey} to set weight")
        sleep(1)
    except KeyboardInterrupt:
        break