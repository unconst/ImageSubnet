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

from protocol import *
import random
from PIL import Image
import io
from typing import List
import json
from config import config
import websocket
import uuid
import urllib
import imagehash
import bittensor as bt
import torchvision.transforms as transforms
import os
import requests
# this object is downloaded from comfyui "Save (API Format)", must enable dev mode
# open prompts/text_to_image.txt
print(config)
server_address = config.comfyui.address + ":" + str(config.comfyui.port)
client_id = str(uuid.uuid4())
print(server_address)

# returns list of images
def t2i(synapse: TextToImage) -> List[Image.Image]:
    prompt = synapse.text
    negative_prompt = synapse.negative_prompt
    height = synapse.height
    width = synapse.width
    num_images_per_prompt = synapse.num_images_per_prompt

    if synapse.seed == -1:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = synapse.seed

    # If making a custom miner, you can dynamicly change the prompt file
    # allow you to run different workflows on the fly
    # by default, we just use the base comfyui workflow
    api_file = 'text_to_image'

    # get directory of this file
    # __file__ is the current file
    #  is the directory of the current file

    with open(f'{os.path.dirname(__file__)}/workflows/{api_file}.txt', 'r') as f:
        TEXT_TO_IMAGE_API = f.read()
    
    # switch based on api_file
    if api_file == 'text_to_image':
        api = json.loads(TEXT_TO_IMAGE_API)
        # api["4"]["inputs"]["ckpt_name"] = "brixlAMustInYour_v20Banu.safetensors" # Used to over ride and set model name
        api["6"]["inputs"]["text"] = prompt
        api["7"]["inputs"]["text"] = negative_prompt
        api["3"]["inputs"]["seed"] = seed
        api["3"]["inputs"]["steps"] = 25
        api["5"]["inputs"]["batch_size"] = num_images_per_prompt
        api["5"]["inputs"]["width"] = width
        api["5"]["inputs"]["height"] = height

    else:
        # custom miners could define other api files using the same format
        raise NotImplementedError
    
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    response = get_images(ws, api)
    ws.close()

    images = []

    for node_id in response:
        for image_data in response[node_id]:
            image = Image.open(io.BytesIO(image_data))
    
            if image.size[0] != width or image.size[1] != height:
                image = image.resize((width, height), Image.ANTIALIAS)

            images.append(image)

    return images

def i2i(synapse: ImageToImage) -> List[Image.Image]:
    print("inside image 2 image")
    prompt = synapse.text
    negative_prompt = synapse.negative_prompt
    height = synapse.height
    width = synapse.width
    num_images_per_prompt = synapse.num_images_per_prompt

    if synapse.seed == -1:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = synapse.seed
    # right now, image is bt.Tensor, we need to deserialize it and convert to PIL image
    image = bt.Tensor.deserialize(synapse.image)
    pil_img =  transforms.ToPILImage()( image )
    
    # generate hash for image using dhash
    hash = imagehash.dhash(pil_img)
    hash = str(hash)

    # generate uid from hash
    uid = uuid.uuid5(uuid.NAMESPACE_DNS, hash)
    uid = str(uid)

    # upload image to comfyui
    # save image to tmp file
    png_name = f'/tmp/{uid}.png'
    pil_img.save(png_name)

    # Define the file to upload
    files = {
        'image': (f'{uid}.png', open(png_name, 'rb'), 'image/png')
    }

    # Define any additional data fields if needed
    data = {
        'name': f'{uid}.png',
        'subfolder': "",
        'type': 'input'
    }

    # Send the POST request with the file and data
    r = requests.post(f'http://{config.comfyui.host}:{config.comfyui.port}/upload/image', files=files, data=data)


    # delete tmp file
    os.remove(png_name)

    # the above errors, a bytes-like object is required, not 'set'
    # the reason is because the file is not being read correctly
    # the file is being read as a set, not a buffered reader
    # the buffered reader is needed to read the file as bytes



    if r.status_code != 200:
        raise Exception("Error uploading image to comfyui")
    
    api_file = 'image_to_image'
    
    with open(f'{os.path.dirname(__file__)}/workflows/{api_file}.txt', 'r') as f:
        IMAGE_TO_IMAGE_API = f.read()
    
    # switch based on api_file
    if api_file == 'image_to_image':
        api = json.loads(IMAGE_TO_IMAGE_API)
        api["6"]["inputs"]["text"] = prompt
        api["7"]["inputs"]["text"] = negative_prompt
        api["3"]["inputs"]["seed"] = seed

        denoise = {"low": 0.3, "medium": 0.7, "high": 0.9}.get(synapse.similarity, 0.0)

        api["3"]["inputs"]["denoise"] = denoise
        api["3"]["inputs"]["steps"] = 25
        api["10"]["inputs"]["image"] = "{}.png".format(uid)
        api["12"]["inputs"]["amount"] = int(num_images_per_prompt)


    else:
        # custom miners could define other api files using the same format
        raise NotImplementedError
    
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    response = get_images(ws, api)
    ws.close()

    images = []

    for node_id in response:
        for image_data in response[node_id]:
            image = Image.open(io.BytesIO(image_data))
    
            if image.size[0] != width or image.size[1] != height:
                image = image.resize((width, height), Image.ANTIALIAS)

            images.append(image)


    return images
    


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images