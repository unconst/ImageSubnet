# queries local miner for images

import bittensor as bt
from protocol import TextToImage, ImageToImage
import asyncio
import torchvision.transforms as transforms
import time
from PIL import Image
from dendrite import AsyncDendritePool
from fabric.utils import tile_images
import os

import ImageReward as RM
scoring_model = RM.load("ImageReward-v1.0", device="cuda")

wallet = bt.wallet()
dendrite = bt.dendrite(wallet=wallet)
bt.trace()


# metagraph
metagraph = bt.metagraph(5, network="finney")
axons = metagraph.axons

# myaxonid = -1 # set this to be your axon id
# if myaxonid == -1:
#     print("Please set myaxonid in test.py:19 to be your axon id")
#     exit()

query = TextToImage(
    text="emotional photo of dancing  gypsy emotional   woman Deidre in rugged ornamental clothes, gypsy ornate headband, dark suntanned skin, long messy hair, native ethnic gold necklace, face, perfect teeth, holding red apple, 8k uhd, high quality,  film grain, looking at viewer, portrait, (skin pores:1.2), (moles:0.8), (imperfect skin:1.1), intricate details, goosebumps, flawless face, ((photorealistic):1.1), (raw, 8k:1.2), hyper realistic, HDR, cinematic, dark, muted colors, atmosphere, ((macro lens)), after dusk, outdoors <lora:entropy-alpha:0.19>",
    negative_prompt="worst quality, nsfw, xxx",
    width=1280,
    height=1280,
    # height=1024*1.5,
    num_images_per_prompt=1,
    seed=-1
)

pool = AsyncDendritePool(wallet, metagraph)

transform = transforms.Compose([
    transforms.PILToTensor()
])

# call_single_uid = dendrite(
#     axons[myaxonid],
#     synapse=query,
#     timeout=30.0
# )

uids = metagraph.uids.tolist() 

# Select up to dendrites_per_query random dendrites.
dendrites_to_query = uids

# await call_single_uid

async def query_async(call_single_uid):
    responses = await pool.async_forward(
        uids = dendrites_to_query,
        query = query,
        timeout = 60
    )
    x = responses

    # create a grid of images and save as one image with tile_images

    _imgs = []

    for j, response in enumerate(x):
        try:
            for image in x[j].images:
                # Convert the raw tensor from the Synapse into a PIL image and display it.
                _imgs.append(transforms.ToPILImage()( bt.Tensor.deserialize(image) ))
            if(len(x[j].images) > 0):
                print("success in " + str(j))
        except Exception as e:
            print(e)
            print("error in " + str(j))
            print(response)
            print(type(response.images[0]))
            # save the response to a file
            # with open("./outputs_bad_responses/" + str(time.time()) + ".json", "w") as f:
            #     f.write(str(response))
                # this causes the tensors to be written as Tensor[3, 512, 512] instead of the actual tensor
                # f.write(str(response)) same thing
            pass
    if(len(_imgs) > 0):
        # scale _imgs down by 0.5
        # _imgs = [ img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS) for img in _imgs ]
        # split _imgs into chunks of 10 or less
        print(len(_imgs), "len imgs")
        chunks = [ _imgs[i:i + 10] for i in range(0, len(_imgs), 10) ]
        total_scoring = []
        for __imgs in chunks:
            ranking, scoring = scoring_model.inference_rank(query.text, __imgs)
            if(type(scoring) == float):
                scoring = [scoring]
            total_scoring += scoring
        print(len(total_scoring))

        # zip scoring and imgs
        _imgs = list(zip(total_scoring, _imgs))
        # sort by scoring
        _imgs.sort(key=lambda x: x[0], reverse=True)

         # top 20 scores
        X = min(20, len(_imgs))
        _scores = list(zip(*_imgs))[0][:X]

        # unzip
        _imgs_unzipped = list(zip(*_imgs))[1]
       

        # avg score
        avg_score = sum(_scores) / len(_scores)
        print(avg_score)

         #  save in /outputs/{prompt with all spaces as _ and other characters removed and only first 12 chars}
        out_dir = "./outputs/" + query.text.replace(" ", "_").replace(",", "").replace(".", "")[:30] + "/"
        # make out dir if not exists
        os.makedirs(out_dir, exist_ok=True)
        # save each image individually under a sub folder out_dir/imgs/ create if needed
        imgs_dir = out_dir + "imgs/"
        os.makedirs(imgs_dir, exist_ok=True)
        for i in range(len(_imgs_unzipped)):
            _imgs_unzipped[i].save(imgs_dir + str(i) + ".png")

        # resize all imgs to half size
        _imgs_unzipped = [ img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS) for img in _imgs_unzipped ]
        # # random name
        _time = time.time()
        name = str(_time)+ f"-{avg_score}".replace(".","_") + ".png"
        tile_images(_imgs_unzipped).save(out_dir + name)

        # save best image outside of imgs/ as best-{_time}.png
        _imgs_unzipped[0].save(out_dir + f"best={_time}.png")

        print("saved best")


        # get the params object first
        params = {
            'text': query.text,
            'negative_prompt': query.negative_prompt,
            'height': query.height,
            'width': query.width,
            'num_images_per_prompt': query.num_images_per_prompt,
            'seed': query.seed,
            'nsfw_allowed': query.nsfw_allowed,
        
        }

        print("init i2i")
        # Update the parameters with the provided kwargs
        try:
            serialized = bt.Tensor.serialize(transform(_imgs_unzipped[0]))
            params.update({
                'image': serialized,
                'similarity': "low"
            })

            # proceed to do image to image with best image
            i2i_query = ImageToImage(**params)

            i2i_responses = await pool.async_forward(
                uids = dendrites_to_query,
                query = i2i_query,
                timeout = 60
            )
        except Exception as e:
            print(e)
            print("error in i2i")
            print(response)
            print(type(response.images[0]))
            # save the response to a file
            # with open("./outputs_bad_responses/" + str(time.time()) + ".json", "w") as f:
            #     f.write(str(response))
                # this causes the tensors to be written as Tensor[3, 512, 512] instead of the actual tensor
                # f.write(str(response)) same thing
            raise e
            
        print("done i2i")
        _imgs = []

        for j, response in enumerate(i2i_responses):
            try:
                for image in i2i_responses[j].images:
                    # Convert the raw tensor from the Synapse into a PIL image and display it.
                    _imgs.append(transforms.ToPILImage()( bt.Tensor.deserialize(image) ))
                if(len(i2i_responses[j].images) > 0):
                    print("success in i2i" + str(j))
            except Exception as e:
                print(e)
                print("error in " + str(j))
                print(response)
                print(type(response.images[0]))
                # save the response to a file
                # with open("./outputs_bad_responses/" + str(time.time()) + ".json", "w") as f:
                #     f.write(str(response))
                    # this causes the tensors to be written as Tensor[3, 512, 512] instead of the actual tensor
                    # f.write(str(response)) same thing
                pass
        print("done looping images")
        print(len(_imgs), "len imgs")
        if(len(_imgs) > 0):
            # scale _imgs down by 0.5
            # _imgs = [ img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS) for img in _imgs ]
            # split _imgs into chunks of 10 or less
            try:
                chunks = [ _imgs[i:i + 10] for i in range(0, len(_imgs), 10) ]
                total_scoring = []
                for __imgs in chunks:
                    ranking, scoring = scoring_model.inference_rank(query.text, __imgs)
                    if(type(scoring) == float):
                        scoring = [scoring]
                    total_scoring += scoring
                print(len(total_scoring))

                # zip scoring and imgs
                _imgs = list(zip(total_scoring, _imgs))
                # sort by scoring
                _imgs.sort(key=lambda x: x[0], reverse=True)

                # top 20 scores
                X = min(20, len(_imgs))
                _scores = list(zip(*_imgs))[0][:X]

                # unzip
                _imgs_unzipped = list(zip(*_imgs))[1]
            

                # avg score
                avg_score = sum(_scores) / len(_scores)
                print(avg_score)

                #  save in /outputs/{prompt with all spaces as _ and other characters removed and only first 12 chars}
                out_dir = "./outputs-i2i/" + query.text.replace(" ", "_").replace(",", "").replace(".", "")[:30] + "/"
                # make out dir if not exists
                os.makedirs(out_dir, exist_ok=True)
                # save each image individually under a sub folder out_dir/imgs/ create if needed
                imgs_dir = out_dir + "imgs/"
                os.makedirs(imgs_dir, exist_ok=True)
                for i in range(len(_imgs_unzipped)):
                    _imgs_unzipped[i].save(imgs_dir + str(i) + ".png")

                # resize all imgs to half size
                _imgs_unzipped = [ img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS) for img in _imgs_unzipped ]
        

                # # random name
                _time = time.time()
                name = str(_time)+ f"-{avg_score}".replace(".","_") + ".png"
                tile_images(_imgs_unzipped).save(out_dir + name)



                # save best image outside of imgs/ as best-{_time}.png
                _imgs_unzipped[0].save(out_dir + f"best-{_time}.png")
            except Exception as e:
                print("got dumb error")
                print(e)
                raise e

        



    # def i2i(t2i: TextToImage, **kwargs) -> ImageToImage:
    #     params = {
    #         'text': t2i.text,
    #         'negative_prompt': t2i.negative_prompt,
    #         'height': t2i.height,
    #         'width': t2i.width,
    #         'num_images_per_prompt': t2i.num_images_per_prompt,
    #         'seed': 696969,
    #     }
        
    #     # Update the parameters with the provided kwargs
    #     params.update(kwargs)

    #     query = ImageToImage(**params)

    #     call_single_uid = dendrite(
    #         axons[myaxonid],
    #         synapse=query,
    #         timeout=30.0
    #     )

    #     queried_async = asyncio.run(query_async(call_single_uid))

    #     return queried_async[0]


    # def show_images(i2i_result: ImageToImage) -> None:
    #     for image in i2i_result.images:
    #         # Convert the raw tensor from the Synapse into a PIL image and display it.
    #         transforms.ToPILImage()( bt.Tensor.deserialize(image) ).show()

    # show_images(i2i(query, image=image, similarity="high", text="an (anime:1.2) woman walking on a path in an autumn forest"))
    # show_images(i2i(query, image=image, similarity="medium", text="an (anime:1.2) woman walking on a path in an autumn forest"))
    # show_images(i2i(query, image=image, similarity="low", text="an (anime:1.2) woman walking on a path in an autumn forest"))

while True:
    asyncio.run(query_async(dendrite))
    time.sleep(1)
    exit()