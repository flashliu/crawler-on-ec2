import json
import os
import asyncio
from urllib.parse import urlparse
from fastapi import APIRouter
import requests
from common import utils
from models.scrap_list_info import ScrapListInfo

router = APIRouter(tags=["Scrap api"])


@router.post("/scrap/list/json")
async def scrapListJson(info: ScrapListInfo):
    url = info.url
    domain = urlparse(url).netloc

    # 从环境变量获取代理
    proxy = os.getenv("PROXY")
    proxies = None

    if proxy:
        print(f"使用代理 {proxy}")
        proxies = {"http": proxy, "https": proxy}
    else:
        proxies = None

    if info.payload is None:
        response = requests.get(info.url, proxies=proxies, verify=False)
    else:
        if info.payload_type == "form":
            # 将 payload 转换为符合 multipart/form-data 的格式
            files = {key: (None, value) for key, value in info.payload.items()}
            response = requests.post(
                info.url, files=files, proxies=proxies, verify=False
            )
        elif info.payload_type == "json":
            response = requests.post(
                info.url, json=info.payload, proxies=proxies, verify=False
            )
    res = response.json()
    s3_uuid = utils.upload_html_to_s3(json.dumps(res))
    list, _ = utils.find_most_related_array(res)

    if list is None:
        return {"listings": [], "parent": None}

    print(f"Detected {len(list)} Watches-----------------")

    tasks = [
        utils.extractWithOpenAI(
            f"Try extract the watch data from the following JSON: "
            + json.dumps(item)
            + "\n\n"
            + f"""
                        There are a few conditions you have to follow:
                        1, the expected fields are name, image, description, brand, price, collection, reference, url(with domain:{domain})
                        2, if one of the fields is missing, set it to null
                        3, don't give me the code, just give me the json result, no need for more explanation
                        4, The image and url must be a full address starting with http or https
                    """,
            model="gpt-4",
        )
        for item in list
    ]
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)

    # 过滤 None 结果并解析 JSON
    output = [json.loads(extracted) for extracted in results if extracted is not None]

    print(f"Found {len(output)} Listing-----------------")

    return {"listings": output, "parent": None, "s3_uuid": s3_uuid}
