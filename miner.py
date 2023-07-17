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
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://test.finney.opentensor.ai')
parser.add_argument('--netuid', type=int, default=64)

config = bt.config( parser )
subtensor = bt.subtensor( 64, config=config )


# Stable diffusion
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import torchvision.transforms as transforms
from protocol import ImageToImage

bt.logging.trace("Loading model: {}".format(config.miner.model))

model_path = config.miner.model
# Lets instantiate the stable diffusion model.
if model_path.endswith('.safetensors') or model_path.endswith('.ckpt') or model_path.startswith('http'):
    # Load from local file or from url.
    model = StableDiffusionPipeline.from_ckpt( model_path, torch_dtype=torch.float16 ).to( config.device )
else:
    # Load from huggingface model hub.
    model =  StableDiffusionPipeline.from_pretrained( model_path , custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16 ).to( config.device )

bt.logging.trace

img2img = StableDiffusionImg2ImgPipeline(**model.components)


transform = transforms.Compose([
    transforms.PILToTensor()
])

async def f( synapse: ImageToImage ) -> ImageToImage:

    seed = synapse.seed

    # Let's set a seed for reproducibility.
    if(seed == -1):
        seed = torch.randint(1000000000, (1,)).item()

    generator = torch.Generator(device=config.device).manual_seed(seed)

    # Check if the batch size is valid.
    if synapse.num_images_per_prompt > config.miner.max_batch_size:
        raise ValueError(f"num_images_per_prompt ({synapse.num_images_per_prompt}) must be less than or equal to max_batch_size ({config.miner.max_batch_size})")

    if synapse.image is not None:
        # If we are doing image to image, we need to use a different pipeline.
        output = img2img(
            image = synapse.image,
            num_images_per_prompt = synapse.num_images_per_prompt,
            num_inference_steps = synapse.num_inference_steps,
            guidance_scale = synapse.guidance_scale,
            negative_prompt = synapse.negative_prompt,
            generator = generator
        )
    else:
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

    synapse.images = []
    for image in output.images:
        img_tensor = transform(image)
        synapse.images.append( bt.Tensor.serialize( img_tensor ) )

    return synapse

def b( synapse: ImageToImage ) -> bool:
    return False

def p( synapse: ImageToImage ) -> float:
    return 0.0

def v( synapse: ImageToImage ) -> None:
    pass

wallet = bt.wallet( config=config )
axon = bt.axon( config, wallet=wallet).attach( f, b, p, v ).start()

# serve axon
subtensor.serve_axon( axon=axon, netuid=config.netuid )

# keep process alive
bt.logging.trace('Miner running. ^C to exit.')

while True:
    try:
        sleep(1)
    except KeyboardInterrupt:
        break