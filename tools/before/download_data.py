import aiohttp
import asyncio
import argparse
import os
import tqdm

from local_lib.utils.file_tools import load_csv_file


def parse_args():
    parser = argparse.ArgumentParser(description='Download data from urls file')
    parser.add_argument('--brand-id', type=int, help='destination root path')
    parser.add_argument('--tag', type=str, help='tag of the dateset')
    parser.add_argument('--task', type=str, help='the task destination root path')
    return parser.parse_args()


async def download_file(url, save_path):
    try:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    with open(save_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
            pbar.update()
    except Exception as e:
        raise Exception(f"Error downloading {url}: {e}")


async def main(download_urls, dst_root, workers=200):
    res = load_csv_file(download_urls, concat_value=True)
    tasks = []

    global semaphore
    semaphore = asyncio.Semaphore(workers)
    for cat, urls in res.items():
        obj_dir = os.path.join(dst_root, cat)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        for url in urls:
            obj_file = os.path.join(obj_dir, os.path.basename(url))
            tasks.append(download_file(url, obj_file))
    
    global pbar
    pbar = tqdm.tqdm(total=len(tasks), desc=f"download img to {tag}_{brand_id}/{task}")
    await asyncio.gather(*tasks, return_exceptions=False)
    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    brand_id = args.brand_id # 1386 #1438 #1267
    task = args.task # "gallery" #"query" # "vis" #
    tag = args.tag
    urls_file = f"{tag}_{brand_id}_{task}.csv"
    assert os.path.exists(urls_file), "please make sure the url file exists!"

    obj_root = f"dataset/function_test/{tag}_{brand_id}/{task}"
    if not os.path.exists(obj_root):
        os.makedirs(obj_root)

    asyncio.run(main(urls_file, obj_root))
