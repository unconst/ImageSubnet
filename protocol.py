

import typing
import pydantic
import bittensor as bt
from typing import Literal

class TextToImage( bt.Synapse ):
    images: list[ bt.Tensor ] = []
    text: str = pydantic.Field( ... , allow_mutation = False)
    negative_prompt: str = pydantic.Field( ... , allow_mutation = False)
    height: int = pydantic.Field( 512 , allow_mutation = False)
    width: int = pydantic.Field( 512 , allow_mutation = False)
    num_images_per_prompt: int = pydantic.Field( 1 , allow_mutation = False)
    seed: int = pydantic.Field( -1 , allow_mutation = False)
    nsfw_allowed: bool = pydantic.Field( False , allow_mutation = False)
    required_hash_fields: list[str] = pydantic.Field( ["text", "negative_prompt", "height", "width", "num_images_per_prompt", "seed", "nsfw_allowed"] , allow_mutation = False)

    # def validate( self ) -> bool:
    #     return len(self.images) == self.num_images_per_prompt and all( [ image.shape[1] == self.height and image.shape[2] == self.width for image in self.images ] ) and all( [ isinstance(image, bt.Tensor) for image in self.images ] )

    # rewrite the above commented function such that it returns True, "" if it is valid, and False, "error message" if it is invalid
    def validate( self ) -> (bool, str):
        # check tensor type first
        if not all( [ isinstance(image, bt.Tensor) for image in self.images ] ):
            return False, "images are not tensors"
        # check number of images
        if len(self.images) != self.num_images_per_prompt:
            return False, "number of images does not match num_images_per_prompt"
        # check image size
        if not all( [ image.shape[1] == self.height and image.shape[2] == self.width for image in self.images ] ):
            return False, "image size does not match height and width"
        # if all checks pass, return True, ""
        return True, ""

class ImageToImage( TextToImage ):
    # Width x height will get overwritten by image size
    image: bt.Tensor = pydantic.Field( ... , allow_mutation = False) 

    # Miners must choose how to define similarity themselves based on their model
    # by default, the strength values are 0.3, 0.7, 0.9
    similarity: Literal["low", "medium", "high"] = pydantic.Field( "medium" , allow_mutation = False) 

    required_hash_fields: list[str] = pydantic.Field(  ["text", "negative_prompt", "height", "width", "num_images_per_prompt", "seed", "nsfw_allowed", "image", "similarity"] , allow_mutation = False)


# TO BE IMPLEMENTED
class Upscale ( TextToImage ): 
    scale: float = pydantic.Field( 2.0 , allow_mutation = False)
