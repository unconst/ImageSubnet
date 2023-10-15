import sys
import os
import ImageReward as RM
from PIL import Image

scoring_model = RM.load("ImageReward-v1.0", device="cuda")

def main():
    # in inputs folder, there should be sub folders
    # each sub folder should contain a file called "prompt.txt" and images .jpg .jpeg .png
    # load all images from each sub folder and compare them to the prompt with ImageReward
    # compare average score, top score, and standard deviation of scores for each sub folder
    # print results to console

    # get list of sub folders
    subFolders = os.listdir("inputs")
    subFolders.sort()

    print("sub folders", subFolders)
    
    # filter out README.md
    if("README.md" in subFolders):
        subFolders.remove("README.md")

    results = []

    # for each sub folder
    for folder in subFolders:
        # get list of images
        images = os.listdir("inputs/" + folder)
        images.sort()

        # if folder empty, pass
        if(len(images) == 0):
            continue

        # throw error if no prompt
        if("prompt.txt" not in images):
            print("Error: no prompt.txt in inputs/" + folder)
            sys.exit(1)

        # get prompt
        prompt = open("inputs/" + folder + "/prompt.txt", "r").read()

        # remove prompt from images
        images.remove("prompt.txt")

        _imgs = [] # PIL.Image list
        # for each image
        for image in images:
            # load image
            img = Image.open("inputs/" + folder + "/" + image)
            _imgs.append(img)
        
        # get scores
        _scores = []
        # split into chunks of 10
        chunks = [ _imgs[i:i + 10] for i in range(0, len(_imgs), 10) ]
        for __imgs in chunks:
            ranking, scoring = scoring_model.inference_rank(prompt, __imgs)
            if(type(scoring) == float):
                scoring = [scoring]
            _scores += scoring
        
        # get average score
        avg_score = sum(_scores) / len(_scores)
        # get top score
        top_score = max(_scores)
        # calculate standard deviation
        std_dev = 0
        for score in _scores:
            std_dev += (score - avg_score) ** 2
        std_dev = (std_dev / len(_scores)) ** 0.5

        # add to results
        results.append({
            "prompt": prompt,
            "avg_score": avg_score,
            "top_score": top_score,
            "std_dev": std_dev,
            "folder": folder
        })

    # sort results by top score
    results.sort(key=lambda x: x["top_score"], reverse=True)

    # print results
    for result in results:
        print(result["folder"] + ":")
        print("  Prompt:", result["prompt"])
        print("  Top Score:", result["top_score"])
        print("  Average Score:", result["avg_score"])
        print("  Standard Deviation:", result["std_dev"])
        print()

if __name__ == "__main__":
    main()