from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
import requests
import bittensor as bt
import sys

transform = transforms.Compose([
    transforms.PILToTensor()
])

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
    # check https://raw.githubusercontent.com/unconst/ImageSubnet/main/VERSION
    # for latest version number
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/unconst/ImageSubnet/main/VERSION"
        )
        response.raise_for_status()
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
                if new__version__ == latest_version:
                    bt.logging.trace("ImageSubnet updated successfully.")
                    bt.logging.trace("Restarting...")
                    bt.logging.trace(f"Running: {sys.executable} {sys.argv}")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    bt.logging.error("ImageSubnet git pull failed you will need to manually update and restart for latest code.")
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
                    print('bad concept', concept_idx, result_img["concept_scores"][concept_idx])
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
