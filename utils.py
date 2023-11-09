from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
import requests
import bittensor as bt
import sys
import argparse
import random
import imagehash

from typing import List
from protocol import TextToImage
from PIL import Image

import torchvision.models as models

import matplotlib.font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument('--no-restart', action="store_true", help='Do not restart after update')

config = bt.config( parser )

# Load prompt dataset.
from datasets import load_dataset
# generate random seed
seed=random.randint(0, 1000000)
dataset = iter(load_dataset("poloclub/diffusiondb")['train'].shuffle(seed=seed).to_iterable_dataset())

# For prompt generation
from transformers import pipeline
prompt_generation_pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")


transform = transforms.Compose([
    transforms.PILToTensor()
])

def get_device(_config: bt.config = None):
    return torch.device(_config.device if (_config is not None and _config.device is not None) else "cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

 # Amount of images
num_images = 1
total_dendrites_per_query = 25
minimum_dendrites_per_query = 3

import ImageReward as RM
scoring_model = None
def get_scoring_model(_config: bt.config = None ):
    global scoring_model
    if scoring_model is None:
        scoring_model = RM.load("ImageReward-v1.0", device=_config.device if (_config is not None and _config.device is not None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    return scoring_model




def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

# load version from VERSION file
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")) as f:
    __version__ = f.read().strip()
    # convert to list of ints
    __version__ = [int(v) for v in __version__.split(".")]

def check_for_updates():
    try:
        bt.logging.trace("Checking for updates...")
        response = requests.get(
            "https://raw.githubusercontent.com/unconst/ImageSubnet/main/VERSION"
        )
        response.raise_for_status()
        try:
            latest_version = response.text.strip()
            latest_version = [int(v) for v in latest_version.split(".")]
            bt.logging.trace(f"Current version: {__version__}")
            bt.logging.trace(f"Latest version: {latest_version}")
            if latest_version > __version__:
                bt.logging.trace("A newer version of ImageSubnet is available. Downloading...")
                # download latest version with git pull
                os.system("git pull")
                # checking local VERSION
                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")) as f:
                    new__version__ = f.read().strip()
                    # convert to list of ints
                    new__version__ = [int(v) for v in new__version__.split(".")]
                    if new__version__ == latest_version and new__version__ > __version__:
                        # run pip install -r requirements.txt silently
                        os.system("pip install -r requirements.txt -q")
                        if not config.no_restart:
                            bt.logging.trace("ImageSubnet updated successfully. Restarting...")
                            bt.logging.trace(f"Running: {sys.executable} {sys.argv}")

                            # add an argument to the end of the command to prevent infinite loop
                            os.execv(sys.executable, [sys.executable] + sys.argv + ["--no-restart"])
                        else:
                            bt.logging.trace("ImageSubnet updated successfully. Restart to apply changes.")
                    else:
                        bt.logging.error("ImageSubnet git pull failed you will need to manually update and restart for latest code.")
        except Exception as e:
            bt.logging.error("Failed to convert response to json: {}".format(e))
            bt.logging.trace("Response: {}".format(response.text))            
    except Exception as e:
        bt.logging.error("Failed to check for updates: {}".format(e))

class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": [], "bad_score": 0.0}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 1.0 # multiplier


            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - (concept_threshold * adjustment), 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - (concept_threshold * adjustment), 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)
                    result_img['bad_score'] += result_img["concept_scores"][concept_idx]

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 and res['bad_score'] > 0.01 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    # images[idx] is a PIL image, so we can't use .shape, convert using transform
                    try:
                        images[idx] = np.zeros(transform(images[idx]).shape)  # black image
                    except:
                        images[idx] = np.zeros((512, 512, 3))

        if any(has_nsfw_concepts):
            print(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts
    
# Determine the rewards based on how close an image aligns to its prompt.
def calculate_rewards_for_prompt_alignment(query: TextToImage, responses: List[ TextToImage ]) -> (torch.FloatTensor, List[ Image.Image ]):

    # Takes the original query and a list of responses, returns a tensor of rewards equal to the length of the responses.
    init_scores = torch.zeros( len(responses), dtype = torch.float32 )
    top_images = []

    for i, response in enumerate(responses):
        # if theres no images, skip this response.
        if len(response.images) == 0:
            top_images.append(None)
            continue

        img_scores = torch.zeros( num_images, dtype = torch.float32 )
        try:
            with torch.no_grad():

                images = []

                for j, tensor_image in enumerate(response.images):
                    # Lets get the image.
                    image = transforms.ToPILImage()( bt.Tensor.deserialize(tensor_image) )

                    images.append(image)
                
                ranking, scores = get_scoring_model(config).inference_rank(query.text, images)
                img_scores = torch.tensor(scores)
                # push top image to images (i, image)
                if len(images) > 1:
                    top_images.append(images[ranking[0]-1])
                else:
                    top_images.append(images[0])
        except Exception as e:
            print(e)
            print("error in " + str(i))
            print(response)
            top_images.append(None)
            continue

        
        # Get the average weight for the uid from _weights.
        init_scores[i] = torch.mean( img_scores )
        #  if score is < 0, set it to 0
        if init_scores[i] < 0:
            init_scores[i] = 0
        
    # if sum is 0 then return empty vector
    if torch.sum( init_scores ) == 0:
        return (init_scores, top_images)

    # preform exp on all values
    init_scores = torch.exp( init_scores )

    # set all values of 1 to 0
    init_scores[init_scores == 1] = 0

    # normalize the scores such that they sum to 1 but skip scores that are 0
    init_scores = init_scores / torch.sum( init_scores )

    return (init_scores, top_images)

def calculate_dissimilarity_rewards( images: List[ Image.Image ] ) -> torch.FloatTensor:
    # Takes a list of images, returns a tensor of rewards equal to the length of the images.
    init_scores = torch.zeros( len(images), dtype = torch.float32 )

    # If array is all nones, return 0 vector of length len(images)
    if all(image is None for image in images):
        return init_scores

    # Calculate the dissimilarity matrix.
    dissimilarity_matrix = compare_to_set(images)

    # Calculate the mean dissimilarity for each image.
    mean_dissimilarities = calculate_mean_similarity(dissimilarity_matrix)

    # Calculate the rewards.
    for i, image in enumerate(images):
        init_scores[i] = mean_dissimilarities[i]

    return init_scores

def get_system_fonts():
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    return font_list

def load_and_preprocess_image_array(image_array, target_size):
    image_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocessed_images = []
    for image in image_array:
        if(image is None):
            preprocessed_images.append(None)
            continue
        image = image_transform(image).unsqueeze(0)
        preprocessed_images.append(image)

    return torch.cat(preprocessed_images, dim=0)

def extract_style_vectors(image_array, target_size=(224, 224)):
    _model = models.vgg19(pretrained=True).features.to(DEVICE)
    _model = nn.Sequential(*list(_model.children())[:35])

    images = load_and_preprocess_image_array(image_array, target_size).to(DEVICE)
    
    with torch.no_grad():
        style_vectors = _model(images)
    style_vectors = style_vectors.view(style_vectors.size(0), -1)
    return style_vectors

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity.item()

def compare_to_set(image_array, target_size=(224, 224)):

    # FUNCTION DESCRIPTION:
    # Calculates the dissimilarity matrix for a set of images.
    # The dissimilarity matrix is a matrix of size (num_images, num_images) where each element is the dissimilarity between two images.
    # The dissimilarity between two images is calculated as 1 - the cosine similarity between the style vectors of the two images.
    # The style vector of an image is the output of the VGG19 network after the image has been passed through the first 35 layers.

    # convert image array to index, image tuple pairs
    image_array = [(i, image) for i, image in enumerate(image_array)]

    # if there are no images, return an empty matrix
    if len(image_array) == 0:
        return []

    # only process images that are not None
    style_vectors = extract_style_vectors([image for _, image in image_array if image is not None], target_size)
    # add back in the None images as zero vectors
    for i, image in image_array:
        if image is None:
            style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)).to(style_vectors.device), style_vectors[i:]))

    dissimilarity_matrix = torch.zeros(len(image_array), len(image_array))
    for i in range(style_vectors.size(0)):
        for j in range(style_vectors.size(0)):
            if image_array[i] is not None and image_array[j] is not None:
                similarity = cosine_similarity(style_vectors[i], style_vectors[j])
                likeness = 1.0 - similarity  # Invert the likeness to get dissimilarity
                likeness = min(1,max(0, likeness))  # Clip the likeness to [0,1]
                if likeness < 0.01:
                    likeness = 0
                dissimilarity_matrix[i][j] = likeness

    return dissimilarity_matrix.tolist()

def calculate_mean_similarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [dissimilarity_matrix[i][j] for j in range(num_images) if i != j]
        # error: list index out of range
        if len(dissimilarity_values) == 0 or sum(dissimilarity_values) == 0:
            mean_dissimilarities.append(0)
            continue
        # divide by amount of non zero values
        non_zero_values = [value for value in dissimilarity_values if value != 0]
        mean_dissimilarity = sum(dissimilarity_values) / len(non_zero_values)
        mean_dissimilarities.append(mean_dissimilarity)

     # Min-max normalization
    non_zero_values = [value for value in mean_dissimilarities if value != 0]

    if(len(non_zero_values) == 0):
        return [0.5] * num_images

    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [(value - min_value) / range_value for value in mean_dissimilarities]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [min(1,max(0, value)) for value in mean_dissimilarities]

    # Ensure sum of values is 1 (normalize)
    # sum_values = sum(mean_dissimilarities)
    # if sum_values != 0:
    #     mean_dissimilarities = [value / sum_values for value in mean_dissimilarities]

    return mean_dissimilarities

def GeneratePrompt():
    global dataset
    
    # Generate a random synthetic prompt. cut to first 20 characters.
    try:
        initial_prompt = next(dataset)['prompt']
    except:
        seed=random.randint(0, 1000000)
        dataset = iter(load_dataset("poloclub/diffusiondb")['train'].shuffle(seed=seed).to_iterable_dataset())
        initial_prompt = next(dataset)['prompt']
    # split on spaces
    initial_prompt = initial_prompt.split(' ')
    # pick a random number of words to keep
    keep = random.randint(1, len(initial_prompt))
    # max of 6 words
    keep = min(keep, 6)
    # keep the first keep words
    initial_prompt = ' '.join(initial_prompt[:keep])
    prompt = prompt_generation_pipe( initial_prompt, min_length=30 )[0]['generated_text']
    return initial_prompt,prompt

def PHashImage(img):
    # check type of input can either be Image.Image, torch.Tensor or bittensor.tensor.Tensor
    if isinstance(img, bt.Tensor):
        img = bt.Tensor.deserialize(img)
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    elif not isinstance(img, Image.Image):
        raise TypeError("img must be of type Image.Image, torch.Tensor or bittensor.tensor.Tensor")
    hash = imagehash.phash( img )
    hash = str(hash)
    return hash