

import typing
import pydantic
import bittensor as bt
from typing import Literal

class TextToImage( bt.Synapse ):
    images: list[ bt.Tensor ] = []
    text: str = pydantic.Field( ... , allow_mutation = False)
    text_hash: str = pydantic.Field( ... , allow_mutation = False)
    negative_prompt: str = pydantic.Field( ... , allow_mutation = False)
    negative_prompt_hash: str = pydantic.Field( ... , allow_mutation = False)
    height: int = pydantic.Field( 512 , allow_mutation = False)
    height_hash: str = pydantic.Field( ... , allow_mutation = False)
    width: int = pydantic.Field( 512 , allow_mutation = False)
    width_hash: str = pydantic.Field( ... , allow_mutation = False)
    num_images_per_prompt: int = pydantic.Field( 1 , allow_mutation = False)
    num_images_per_prompt_hash: str = pydantic.Field( ... , allow_mutation = False)
    seed: int = pydantic.Field( -1 , allow_mutation = False)
    seed_hash: str = pydantic.Field( ... , allow_mutation = False)
    nsfw_allowed: bool = pydantic.Field( False , allow_mutation = False)
    nsfw_allowed_hash: str = pydantic.Field( ... , allow_mutation = False)
    # _t: str = 'TextToImage'
    # images: list[ bt.Tensor ] = []
    # text: str = pydantic.Field( ... , allow_mutation = False)
    # fixed_resolution: bool = pydantic.Field( False , allow_mutation = False) # if true, images are forced to be the exact widthxheight, else they're forced to match the aspect ratio
    # height: int = pydantic.Field( 512 , allow_mutation = False)
    # width: int = pydantic.Field( 512 , allow_mutation = False)
    # num_images_per_prompt: int = pydantic.Field( 1 , allow_mutation = False)
    # num_inference_steps: int = 20
    # guidance_scale: float = 7.5
    # negative_prompt: str = pydantic.Field( ... , allow_mutation = False)
    # seed: int = pydantic.Field( -1 , allow_mutation = False)
    # nsfw_allowed: bool = pydantic.Field( False , allow_mutation = False)

class ImageToImage( TextToImage ):
    image: bt.Tensor = pydantic.Field( ... , allow_mutation = False)
    image_hash: str = pydantic.Field( ... , allow_mutation = False)

    # Miners must choose how to define similarity themselves based on their model
    # by default, the strength values are 0.3, 0.7, 0.9
    similarity: Literal["low", "medium", "high"] = pydantic.Field( "medium" , allow_mutation = False) 
    similarity_hash: str = pydantic.Field( ... , allow_mutation = False)



# TO BE IMPLEMENTED
class Upscale ( TextToImage ): 
    scale: float = pydantic.Field( 2.0 , allow_mutation = False)
