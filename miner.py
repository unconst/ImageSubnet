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

required_version = (3, 9)

if sys.version_info < required_version:
    raise Exception(f"Python version {required_version[0]}.{required_version[1]} or higher is required.")

# Imports
import torch
import typing
import pydantic
import bittensor as bt
import argparse
from time import sleep

bt.debug()
bt.trace()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--miner.model', type=str, default='prompthero/openjourney-v4')
parser.add_argument('--miner.max_batch_size', type=int, default=4)
parser.add_argument('--miner.allow_nsfw', type=bool, default=False)
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://test.finney.opentensor.ai')
parser.add_argument('--wallet.hotkey', type=str, default='default')
parser.add_argument('--wallet.name', type=str, default='default')
parser.add_argument('--wallet.path', type=str, default='~/.bittensor/wallets')
parser.add_argument('--netuid', type=int, default=64)
parser.add_argument('--axon.port', type=int, default=3000)

config = bt.config( parser )
subtensor = bt.subtensor( 64, config=config, chain_endpoint=config.subtensor.chain_endpoint )


from utils import StableDiffusionSafetyChecker, CLIPImageProcessor
bt.logging.trace("Loading safety checker")
safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( config.device )
processor = CLIPImageProcessor()

if config.miner.allow_nsfw:
    bt.logging.warning("NSFW is enabled. Without a filter, your miner may generate unwanted images. Please use with caution.")

# Stable diffusion
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import torchvision.transforms as transforms
from protocol import TextToImage

bt.logging.trace("Loading model: {}".format(config.miner.model))

model_path = config.miner.model
# Lets instantiate the stable diffusion model.
if model_path.endswith('.safetensors') or model_path.endswith('.ckpt'):
    # Load from local file or from url.
    model = StableDiffusionPipeline.from_ckpt( model_path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False ).to( config.device )
else:
    # Load from huggingface model hub.
    model =  StableDiffusionPipeline.from_pretrained( model_path , custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False ).to( config.device )

bt.logging.trace

img2img = StableDiffusionImg2ImgPipeline(**model.components)


transform = transforms.Compose([
    transforms.PILToTensor()
])

async def f( synapse: TextToImage ) -> TextToImage:

    seed = synapse.seed

    # Let's set a seed for reproducibility.
    if(seed == -1):
        seed = torch.randint(1000000000, (1,)).item()

    generator = torch.Generator(device=config.device).manual_seed(seed)

    # Check if the batch size is valid.
    if synapse.num_images_per_prompt > config.miner.max_batch_size:
        raise ValueError(f"num_images_per_prompt ({synapse.num_images_per_prompt}) must be less than or equal to max_batch_size ({config.miner.max_batch_size})")

    output = GenerateImage(synapse, generator)

    synapse.images = []

    has_nsfw_concept = CheckNSFW(output, synapse)
    if any(has_nsfw_concept):
        output.images = [image for image, has_nsfw in zip(output.images, has_nsfw_concept) if not has_nsfw]
        # try to regenerate another image once
        copy_synapse = synapse.copy()
        copy_synapse.num_images_per_prompt = 1
        output2 = GenerateImage(copy_synapse, generator)
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
        clip_input = processor([transform(image) for image in output.images], return_tensors="pt").to( config.device )
        images, has_nsfw_concept = safetychecker.forward( images=output.images, clip_input=clip_input.pixel_values.to( config.device ))
        return has_nsfw_concept
    else:
        return [False] * len(output.images)

def GenerateImage(synapse, generator):
    try:
        # If we are doing image to image, we need to use a different pipeline.
        output = img2img(
            image = synapse.image,
            num_images_per_prompt = synapse.num_images_per_prompt,
            num_inference_steps = synapse.num_inference_steps,
            guidance_scale = synapse.guidance_scale,
            negative_prompt = synapse.negative_prompt,
            generator = generator
        )
    except:
        output = model(
            prompt = synapse.text,
            height = synapse.height,
            width = synapse.width,
            num_images_per_prompt = synapse.num_images_per_prompt,
            num_inference_steps = synapse.num_inference_steps,
            guidance_scale = synapse.guidance_scale,
            negative_prompt = synapse.negative_prompt,
            generator = generator
        )
        
    return output

def b( synapse: TextToImage ) -> bool:
    return False

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
        sleep(1)
    except KeyboardInterrupt:
        break