# queries local miner for images

import bittensor as bt
from protocol import TextToImage, ImageToImage
import asyncio
import torchvision.transforms as transforms
import time
from PIL import Image

wallet = bt.wallet()
dendrite = bt.dendrite(wallet=wallet)
bt.trace()


# metagraph
# metagraph = bt.metagraph(64, network="test")
# axons = metagraph.axons

axon = bt.axon(port=8081)

query = TextToImage(
    text="an (anime:1.2) beautiful autumn forest scenery with the wind blowing the leaves",
    negative_prompt="worst quality, nsfw, xxx",
    width=768,
    height=768,
    num_images_per_prompt=1,
    seed=-1
)

call_single_uid = dendrite(
    axon,
    synapse=query,
    timeout=30.0
)


# await call_single_uid

async def query_async(call_single_uid):
    corutines = [call_single_uid]
    return await asyncio.gather(*corutines)

x = asyncio.run(query_async(call_single_uid))

for image in x[0].images:
    # Convert the raw tensor from the Synapse into a PIL image and display it.
    transforms.ToPILImage()( bt.Tensor.deserialize(image) ).show()
def i2i(t2i: TextToImage, **kargs) -> ImageToImage:
    query = ImageToImage(
        text=t2i.text,
        negative_prompt=t2i.negative_prompt,
        height=t2i.height,
        width=t2i.width,
        num_images_per_prompt=t2i.num_images_per_prompt,
        seed=696969,
        **kargs,
    )

    call_single_uid = dendrite(
        bt.axon(port=8081),
        synapse=query,
        timeout=30.0
    )

    queried_async = asyncio.run(query_async(call_single_uid))

    return queried_async[0]


def show_images(i2i_result: ImageToImage) -> None:
    for image in i2i_result.images:
        # Convert the raw tensor from the Synapse into a PIL image and display it.
        transforms.ToPILImage()( bt.Tensor.deserialize(image) ).show()

show_images(i2i(query, image=image, similarity="high"))
show_images(i2i(query, image=image, similarity="medium"))
show_images(i2i(query, image=image, similarity="low"))