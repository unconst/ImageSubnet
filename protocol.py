

import typing
import pydantic
import bittensor as bt

class TextToImage( bt.Synapse ):
    image: typing.Optional[ bt.Tensor ] = None
    text: str = pydantic.Field( ..., allow_mutation = False)
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1 
    num_inference_steps: int = 50
    guidance_scale: float = 7.5 
    negative_prompt: str = ""
    seed: int = -1