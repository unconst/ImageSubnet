

import typing
import pydantic
import bittensor as bt
from typing import Union, TypeVar

class TextToImage( bt.Synapse ):
    _t: str = 'TextToImage'
    images: list[ bt.Tensor ] = []
    text: str = pydantic.Field( ... , allow_mutation = False)
    fixed_resolution: bool = pydantic.Field( False , allow_mutation = False) # if true, images are forced to be the exact widthxheight, else they're forced to match the aspect ratio
    height: int = pydantic.Field( 512 , allow_mutation = False)
    width: int = pydantic.Field( 512 , allow_mutation = False)
    num_images_per_prompt: int = pydantic.Field( 1 , allow_mutation = False)
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = pydantic.Field( ... , allow_mutation = False)
    seed: int = pydantic.Field( -1 , allow_mutation = False)
    nsfw_allowed: bool = pydantic.Field( False , allow_mutation = False)

class ImageToImage( TextToImage ):
    _t: str = 'ImageToImage'
    image: bt.Tensor = pydantic.Field( ... , allow_mutation = False)
    strength: float = pydantic.Field( 0.75 , allow_mutation = False)

T = TypeVar('T', bound=Union[TextToImage, ImageToImage])