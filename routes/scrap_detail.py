import io
import json
import asyncio
from urllib.parse import urlparse
from fastapi import APIRouter, Body
from common import utils
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

router = APIRouter(tags=["Scrap api"])

executor = ThreadPoolExecutor(max_workers=3)


def run_selenium_scraping(url: str):
    try:
        driver, temp_dirs = utils.get_driver()
        driver.get(url)
        utils.wait_for_requests_to_complete(driver)
        utils.handle_popup(driver)

        driver.set_window_size(1920, 2000)

        text = driver.execute_script("return document.body.innerText")
        screenshot = driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        watch_boxes = utils.watch_detect(image, threshold=0.05)

        images_html = utils.get_detail_images_html(watch_boxes, driver)
        return text, images_html, None
    except Exception as e:
        print(e)
        return None, None, {"error": str(e)}
    finally:
        utils.clean_up_driver(driver, temp_dirs)
        # 释放内存
        image.close()
        del image
        del screenshot
        del text
        del temp_dirs
        del watch_boxes
        driver.quit()


@router.post("/scrap/detail")
async def scrapDetail(url: str = Body(..., embed=True)):
    loop = asyncio.get_event_loop()
    text, images_html, error = await loop.run_in_executor(
        executor, run_selenium_scraping, url
    )

    if error is not None:
        return error
        
    domain = urlparse(url).netloc
    res = await utils.extractWithOpenAI(
        "Extract the watch data, expect fields:brand,collection,reference,price, return a json, just give me single layer json result, the price should be with currency symbol.from the following text:"
        + text,
        model="gpt-4",
    )

    result = json.loads(res)

    if images_html is not None:
        images = await utils.extractWithOpenAI(
            f"Extract the image urls,just give me a json url(with domain:{domain}) list,Remove identical images and keep the largest size, from the following html:"
            + images_html
        )
        if images is not None:
            result["images"] = json.loads(images)

    return result
