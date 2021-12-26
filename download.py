import numpy as np
import glob
import pandas as pd
import asyncio 
import aiohttp
import aiofiles
import time 
from concurrent.futures import ThreadPoolExecutor

"""
Collects photo urls from the unsplash-lite dataset
"""
path = './unsplash-lite/photos.tsv*'
files = glob.glob(path)
subsets = []
for filename in files:
    df = pd.read_csv(filename, sep='\t', header=0)
    subsets.append(df)

image_info = pd.concat(subsets, axis=0, ignore_index = True)

"""
Create function to download image from urls
"""
async def get_images():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        for i in range(len(image_info)):
            try: 
                file_name = "./images/img" + str(i) + ".jpg"
                img_url = image_info.iloc[i]['photo_image_url'] + "?w=700"
                response = await session.get(img_url)
                if response.status == 200:
                    f = await aiofiles.open(file_name, mode='wb')
                    await f.write(await response.read())
                    await f.close()
                print("Img" + str(i) + " downloaded...")
            except:
                pass

    end = time.time()
    print(end - start)

"""
Create asynchronous with multiprocessing event loop
"""
loop = asyncio.get_event_loop()
p = ThreadPoolExecutor(128) 
loop.run_until_complete(get_images())


"""
Time to download will vary depending on latency and network speed.
"""