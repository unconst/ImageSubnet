import bittensor as bt
from protocol import TextToImage
from typing import List
import asyncio
import torchvision.transforms as transforms

class AsyncDendritePool:
    def __init__(self, wallet, metagraph):
        self.metagraph = metagraph
        self.dendrite = bt.dendrite(wallet=wallet)
    
    async def async_forward(
            self,
            uids: List[int],
            query: TextToImage,
            timeout: float = 12.0
    ):

        def call_single_uid(uid):
            query_copy = query.copy()
            return self.dendrite(
                self.metagraph.axons[uid],
                synapse=query_copy,
                timeout=timeout
            )

        
        async def query_async():
            corutines = [call_single_uid(uid) for uid in uids]
            return await asyncio.gather(*corutines)
        
        return await query_async()