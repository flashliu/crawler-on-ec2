import asyncio
import base64
import io
import json
import logging
import gc
from urllib.parse import urlparse
from fastapi import APIRouter
from common import utils
from PIL import Image
from time import sleep
from selenium.webdriver.common.by import By
from concurrent.futures import ProcessPoolExecutor

from models.scrap_list_browser_info import ScrapListBrowserInfo

router = APIRouter(tags=["Scrap api"])

def run_selenium_scraping(info: ScrapListBrowserInfo):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("seleniumwire").setLevel(logging.ERROR)
    logging.getLogger("WDM").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("watchfiles.watcher").setLevel(logging.ERROR)

    driver, temp_dirs = utils.get_driver()
    url = info.url
    logging.info(f"Scrap with browser: {url}")

    try:
        driver.get(url)
        sleep(5)
        utils.wait_for_requests_to_complete(driver)
        popups = utils.handle_popup(driver)
        logging.info(popups)

        driver.execute_script(
            """
            window.scroll({
                top: 99999,
                left: 0,
                behavior: 'smooth'
            });
        """
        )

        utils.wait_for_requests_to_complete(driver)
        utils.wait_for_images_to_load(driver)

        width = driver.execute_script(
            "return Math.max(document.body.scrollWidth, document.body.offsetWidth, document.documentElement.clientWidth, document.documentElement.scrollWidth, document.documentElement.offsetWidth);"
        )
        height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
        )

        logging.info(f"Window size: {width}x{height}")
        driver.set_window_size(width, height)

        driver.execute_script(
            """
            window.scroll({
                top: 0,
                left: 0,
                behavior: 'smooth'
            });
        """
        )

        full_page = driver.find_element(By.TAG_NAME, "body")
        screenshot = full_page.screenshot_as_png

        image = Image.open(io.BytesIO(screenshot))
        watch_boxes = utils.watch_detect(image)

        logging.info(f"Detected {len(watch_boxes)} Watches-----------------")

        if len(watch_boxes) == 0:
            return [], None, driver.page_source, None

        html_list, parent = utils.get_html_list(watch_boxes, driver)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return html_list or [], parent, driver.page_source, image_base64

    except Exception as e:
        logging.info(e)
        return [], None, None, None
    finally:
        utils.clean_up_driver(driver, temp_dirs)
        driver.quit()
        image.close()
        gc.collect()  # 手动触发垃圾回收
        logging.info(f"Quit driver {driver.session_id}")


@router.post("/scrap/list/browser")
async def scrapListBrowser(info: ScrapListBrowserInfo):
    # 在每个请求中创建单独的 ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_event_loop()
        domain = urlparse(info.url).netloc

        try:
            html_list, parent, page_source, image_base64 = await loop.run_in_executor(
                executor, run_selenium_scraping, info
            )

            s3_uuid = await utils.upload_html_to_s3(page_source)

            if len(html_list) == 0:
                return {
                    "message": "Can not find any watches",
                    "listings": [],
                    "s3_uuid": s3_uuid,
                    "parent": None,
                    "image_base64": image_base64,
                }

            if info.parent is not None and parent != info.parent:
                return {"listings": [], "parent": parent, "s3_uuid": s3_uuid}

            if html_list is not None:
                logging.info(f"Found {len(html_list)} DOM elements-----------------")

                tasks = [
                    utils.extractWithOpenAI(
                        f"Try extract the watch data from the following HTML: "
                        + html
                        + "\n\n"
                        + f"""
                            There are a few conditions you have to follow:

                            1, the expected fields are name, image, description, brand, price, collection, reference, url(with domain:{domain})
                            2, if one of the fields is missing, set it to null
                            3, don't give me the code, just give me the json result, no need for more explanation
                            4, The image and url must be a full address starting with http or https
                            5, The price should be with currency symbol
                        """,
                        model="gpt-4",
                    )
                    for html in html_list
                ]

                results = await asyncio.gather(*tasks)

                output = [
                    json.loads(extracted) for extracted in results if extracted is not None
                ]

                return {"listings": output, "parent": parent, "s3_uuid": s3_uuid}

            else:
                return {"listings": [], "parent": parent, "s3_uuid": s3_uuid}
        
        except Exception as e:
            logging.error(f"Error: {e}")
            return {"message": "Error during scraping"}

