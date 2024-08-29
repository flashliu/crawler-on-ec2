import io
import json
from urllib.parse import urlparse
from fastapi import APIRouter, Body
from common import utils
from PIL import Image

router = APIRouter(tags=["Scrap api"])


@router.post("/scrap/detail")
async def scrapDetail(url: str = Body(..., embed=True)):
    try:
        domain = urlparse(url).netloc
        driver, temp_dirs = utils.get_driver()
        driver.get(url)
        utils.wait_for_requests_to_complete(driver)
        utils.handle_popup(driver)

        driver.set_window_size(1920, 2000)

        text = driver.execute_script("return document.body.innerText")

        res = await utils.extractWithOpenAI(
            "Extract the watch data, expect fields:brand,collection,reference,price, return a json, just give me single layer json result, the price should be with currency symbol.from the following text:"
            + text,
            model="gpt-4",
        )

        result = json.loads(res)
        del text

        screenshot = driver.get_screenshot_as_png()
        # 打开截图
        image = Image.open(io.BytesIO(screenshot))
        watch_boxes = utils.watch_detect(image, threshold=0.05)

        # 释放内存
        image.close()
        del image
        del screenshot

        images_html = utils.get_detail_images_html(watch_boxes, driver)
        if images_html is not None:
            images = await utils.extractWithOpenAI(
                f"Extract the image urls,just give me a json url(with domain:{domain}) list,Remove identical images and keep the largest size, from the following html:"
                + images_html
            )
            if images is not None:
                result["images"] = json.loads(images)
            del images
            del images_html

        del watch_boxes

        return result
    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        utils.clean_up_driver(driver, temp_dirs)
        del temp_dirs
        driver.quit();
