

import typing
import pydantic
import bittensor as bt

class TextToImage( bt.Synapse ):

    images: list[ bt.Tensor ]
    text: str = pydantic.Field( ... , allow_mutation = False)
    height: int = pydantic.Field( 512 , allow_mutation = False)
    width: int = pydantic.Field( 512 , allow_mutation = False)
    num_images_per_prompt: int = pydantic.Field( 1 , allow_mutation = False)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = pydantic.Field( ... , allow_mutation = False)
    seed: int = pydantic.Field( -1 , allow_mutation = False)

class ImageToImage( TextToImage ):
    image: bt.Tensor = pydantic.Field( ... , allow_mutation = False)