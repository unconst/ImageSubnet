# ImageSubnet

Image Subnet, or text2image, is built by default to host and run Stable Diffusion models. However, it is adaptable, and any model can be run on the network that takes in a prompt, width, and height paramaters.

The Validator is built such that it ranks miners images on aesthetic and how closely they match the given prompt. Also, images which are too similiar in style will be slightly penalized to promote diverisity among image models hosted on miners.

## Getting Started


**As of writting this (aug 10th), defaults still point to testnet, not mainnet**

Both Validators and Miners will need to register to subnet **41** in order to participate in the ImageSubnet. You can do so by recycle registering like so `btcli recycle_register --netuid 41`


### For **miners** the command you will run is `py miner.py --miner.model [huggingface/repo OR path_to/model.safetensors]`

Optional arguments include

`--device` default: **cuda**, where to run the image model off of

`--miner.model` default: **prompthero/openjourney-v4**, a huggingface repo or local safetensors file

`--miner.max_batch_size` default: **1**, the maximum number of images your miner will generate per request

`--miner.allow_nsfw` default: **Flase**, set to True if you wish to allow NSFW content. *(Warning, this may produce unwanted content)*

`--subtensor.chain_endpoint` default: **mainnet opentensor subnet**, override to use a custom subnet endpoint

`--wallet.name` default: **default**, name of your wallet

`--wallet.hotkey` default: **default**, set wallet hotkey name

`--wallet.path` default: `~/.bittensor/wallets`, the path to which your bittensor wallets reside at

`--netuid` default: **41**, the subnet you want to connect to (64 on testnet)

`--axon.port` default: **3000**, port to launch your axon in, this needs to be open to the public


### For **validators**, the command you will run is `py validator.py`


Optional arguments include

`--subtensor.chain_endpoint` default: **mainnet opentensor subnet**, override to use a custom subnet endpoint

`--validator.allow_nsfw` default: **False**, as a miner, choose to allow NSFW content from miners. *Note: Miners still have the option of refusing to generate nsfw content.*

`--validator.save_images` default: **False**, save images and prompts to `save_dir`

`--validator.save_dir` default: `./images`, path to a folder to save images, folder will be created if it doesnt exist. If a custom path is set `--validator.save_images` will be set to `True`

 `--validator.use_absolute_size` default: **False**, set to True if you want the exact width and height, else if a miner responds with the same aspect ratio they wont be penalized.