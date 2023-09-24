import argparse
import bittensor as bt

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--comfyui.address', type=str, default='127.0.0.1')
parser.add_argument('--comfyui.port', type=int, default=8188)
parser.add_argument('--comfyui.path', type=str, default=None)
parser.add_argument('--miner.allow_nsfw', type=bool, default=False)
parser.add_argument('--miner.model_warning', type=bool, default=True)
parser.add_argument('--miner.max_batch_size', type=int, default=1)
parser.add_argument('--miner.steps.max', type=int, default=50)
parser.add_argument('--miner.steps.min', type=int, default=10)
parser.add_argument('--miner.guidance.max', type=float, default=None)
parser.add_argument('--miner.guidance.min', type=float, default=0)
parser.add_argument('--miner.height.max', type=int, default=2048)
parser.add_argument('--miner.height.min', type=int, default=None)
parser.add_argument('--miner.width.max', type=int, default=2048)
parser.add_argument('--miner.width.min', type=int, default=None)
parser.add_argument('--miner.max_images', type=int, default=4)
parser.add_argument('--miner.max_pixels', type=int, default=None) # determines total number of images able to generate in one batch (height * width * num_images_per_prompt)
parser.add_argument('--subtensor.chain_endpoint', type=str, default='wss://test.finney.opentensor.ai')
parser.add_argument('--wallet.hotkey', type=str, default='default')
parser.add_argument('--wallet.name', type=str, default='default')
parser.add_argument('--wallet.path', type=str, default='~/.bittensor/wallets')
parser.add_argument('--netuid', type=int, default=64)
parser.add_argument('--axon.port', type=int, default=3000)

config = bt.config( parser )

if config.miner.model_warning:
    bt.logging.warning("please check ./comfy/workflows/<your workflow>.txt to ensure the models you use are located inside ComfyUI/models/checkpoints \n You can disable the above warning with --miner.model_warning False")

# warn if there is no comfyui path
if config.comfyui.path is None:
    bt.logging.warning("WARNING: --miner.comfyui.path is not set, image cache will not be cleared automatically")

# check to see if port 8188 is open
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 8188))
if result != 0:
    raise argparse.ArgumentTypeError("ComfyUI is not detected on port 8188, please run ComfyUI and try again. Or to define a custom port use --miner.comfyui.port <port>")

# verify min/max height and width as they should all be divisible by 8
if config.miner.height.max is not None and config.miner.height.max % 8 != 0:
    raise argparse.ArgumentTypeError(f"height.max must be divisible by 8, but got {config.miner.height.max}")
if config.miner.height.min is not None and config.miner.height.min % 8 != 0:
    raise argparse.ArgumentTypeError(f"height.min must be divisible by 8, but got {config.miner.height.min}")
if config.miner.width.max is not None and config.miner.width.max % 8 != 0:
    raise argparse.ArgumentTypeError(f"width.max must be divisible by 8, but got {config.miner.width.max}")
if config.miner.width.min is not None and config.miner.width.min % 8 != 0:
    raise argparse.ArgumentTypeError(f"width.min must be divisible by 8, but got {config.miner.width.min}")
