
from PIL import Image
import imagehash
import bittensor as bt
from protocol import TextToImage, ImageToImage
import argparse
import random
import asyncio
import torchvision.transforms as transforms

# Load the config.
parser = argparse.ArgumentParser()
parser.add_argument( '--netuid', type = int, default = 5 )
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://entrypoint-finney.opentensor.ai')
parser.add_argument('--subtensor._mock', type=bool, default=False)
parser.add_argument('--validator.allow_nsfw', type=bool, default=False)
parser.add_argument('--validator.save_dir', type=str, default='./images')
parser.add_argument('--validator.save_images', type=bool, default=False)
parser.add_argument('--validator.use_absolute_size', type=bool, default=False) # Set to True if you want to 100% match the input size, else just match the aspect ratio
parser.add_argument('--validator.label_images', type=bool, default=False, help="if true, label images with dendrite uid and score")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--axon.port', type=int, default=3000)
parser.add_argument('--uid', type=int, default=None)
parser.add_argument('--prompt', type=str, default="A pink happy unicorn dancing on rainbows")
bt.wallet.add_args( parser )
bt.subtensor.add_args( parser )
config = bt.config( parser )


# if theres no uid exit
if config.uid == None:
    print('Please specify a --uid to query')
    exit()

wallet = bt.wallet( config = config )
sub = bt.subtensor( config = config )
meta = sub.metagraph( config.netuid )
meta.sync( subtensor = sub )
dend = bt.dendrite( wallet = wallet )

seed=random.randint(0, 1000000)

async def main():
    # create the TextToImage
    text_to_image = TextToImage(
        text = config.prompt,
        negative_prompt = "",
        height = 768,
        width = 768,
        num_images_per_prompt = 1,
        seed = seed,
        uid = config.uid
    )

    # call the uid with dendrite 
    corutine = dend(
        meta.axons[config.uid],
        synapse = text_to_image,
        timeout = 20.0
    )
    print("Attempting to call uid: ", config.uid, " with a TextToImage synapse")
    # get the response
    response = await asyncio.gather(*[corutine])

    if len(response[0].images) == 0:
        print("No images found for uid: ", config.uid)
        return

    # get the image
    image = response[0].images[0]

    # the image is in a serialized format, deserialize it
    image = bt.Tensor.deserialize(image)

    # convert to PIL image
    pil_img =  transforms.ToPILImage()( image )

    # generate hash for image using dhash
    hash = imagehash.phash(pil_img)
    hash = str(hash)

    print("hash: ", hash)

    # now do image to image
    image_to_image = ImageToImage(
        text = "",
        negative_prompt = "",
        height = 768,
        width = 768,
        num_images_per_prompt = 1,
        seed = seed,
        image = response[0].images[0],
        uid = config.uid,
        similarity="low"
    )

    # call the uid with dendrite
    corutine = dend(
        meta.axons[config.uid],
        synapse = image_to_image,
        timeout = 20.0
    )

    # wait before calling as to not trigger the time blacklist
    print("waiting 60s before calling uid: ", config.uid, " with a ImageToImage synapse")
    await asyncio.sleep(15)
    print("45s remaining")
    await asyncio.sleep(15)
    print("30s...")
    await asyncio.sleep(15)
    print("15s...")
    await asyncio.sleep(15)

    print("Attempting to call uid: ", config.uid, " with a ImageToImage synapse")

    # get the response
    response = await asyncio.gather(*[corutine])

    if len(response[0].images) == 0:
        print("No images found for uid: ", config.uid, " with a ImageToImage synapse")
        return

    # get the image
    image = response[0].images[0]

    # the image is in a serialized format, deserialize it
    image = bt.Tensor.deserialize(image)

    # convert to PIL image
    pil_img =  transforms.ToPILImage()( image )

    # generate hash for image using phash
    hash = imagehash.phash(pil_img)

    print("hash: ", hash)

    print("Done!")

asyncio.run(main())