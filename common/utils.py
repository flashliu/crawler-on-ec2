import os
import time
import boto3
import uuid
import gc
import glob
import shutil
import re
from bs4 import BeautifulSoup
import torch

from tempfile import mkdtemp
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from seleniumwire import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from collections import Counter
from openai import AsyncAzureOpenAI
from transformers import (
    pipeline,
    OwlViTForObjectDetection,
    OwlViTProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from collections import defaultdict
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from torchvision.ops import box_iou
from selenium.webdriver.common.by import By


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # options.add_argument("--single-process")
    # options.add_argument("--disable-ipv6")
    # options.add_argument("--remote-debugging-port=9222")

    # 创建临时目录并保存路径
    user_data_dir = mkdtemp()
    data_path = mkdtemp()
    disk_cache_dir = mkdtemp()
    homedir = mkdtemp()

    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument(f"--data-path={data_path}")
    options.add_argument(f"--disk-cache-dir={disk_cache_dir}")
    options.add_argument(f"--homedir={homedir}")

    PROXY_URL = os.getenv("PROXY", None)

    seleniumwire_options = {}
    # use proxy if PROXY_URL
    if PROXY_URL:
        seleniumwire_options = {
            "proxy": {"http": f"{PROXY_URL}", "https": f"{PROXY_URL}"},
        }
        print(f"use proxy {PROXY_URL}")
    else:
        print("no proxy")

    service = ChromeService(ChromeDriverManager().install())
    try:
        # 添加延迟确保 Chrome 完全启动
        time.sleep(5)

        driver = webdriver.Chrome(
            service=service, seleniumwire_options=seleniumwire_options, options=options
        )
        driver.set_page_load_timeout(600)
    except Exception as e:
        print(f"Error starting Chrome WebDriver: {e}")
        raise

    # 返回临时目录路径和 driver
    return driver, [user_data_dir, data_path, disk_cache_dir, homedir]


# 清理临时目录和关闭 WebDriver 的方法
def clean_up_driver(driver, temp_dirs):
    try:
        driver.quit()
        print("Chrome WebDriver successfully quit.")
    except Exception as e:
        print(f"Error quitting Chrome WebDriver: {e}")

    for dir_path in temp_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"Temporary directory {dir_path} successfully removed.")
        except Exception as e:
            print(f"Error removing temporary directory {dir_path}: {e}")
    delete_files("/tmp/.pki/*")
    delete_files("/tmp/core.chrome.*")
    # 触发垃圾回收
    gc.collect()
    check_space("After clean_up_driver")
    # get_tmp_files()


def delete_files(temp_files):
    # 匹配 /tmp 目录下所有 core.chrome.* 文件
    files = glob.glob(temp_files)

    # 删除匹配的每一个文件
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted file: {file}")


def get_tmp_files():
    for root, dirs, files in os.walk("/tmp"):
        for file in files:
            print(f"clean_up_driver {os.path.join(root, file)}")


def check_space(title):
    total, used, free = shutil.disk_usage("/tmp")
    print(
        f"{title} Total: {total/1024/1024} MB, Used: {used/1024/1024} MB, Free: {free/1024/1024} MB"
    )


# 启用CDP监听
def enable_request_logging(driver):
    driver.execute_cdp_cmd("Network.enable", {})


# 等待所有请求完成
def wait_for_requests_to_complete(driver, timeout=60):
    start_time = time.time()

    while time.time() - start_time < timeout:
        incomplete_requests = [req for req in driver.requests if req.response is None]

        # 调试输出当前未完成的请求
        if incomplete_requests:
            for req in incomplete_requests:
                print(f"未完成请求: {req.method} {req.url}")

        # 如果没有未完成的请求，则跳出循环
        if not incomplete_requests:
            print("所有请求已完成")
            return

        time.sleep(1)  # 每秒检查一次

    print("等待请求完成超时")


def watch_detect(image, threshold=0.002):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    model_path = os.path.join(current_dir, "owlvit-base-patch32")

    # 加载模型和处理器
    processor = OwlViTProcessor.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path)

    # 使用本地模型初始化 pipeline
    detector = pipeline(
        "zero-shot-object-detection",
        model=model,
        image_processor=image_processor,  # 显式指定图像处理器
        tokenizer=processor.tokenizer,  # 显式指定文本处理器
    )

    results = detector(image, candidate_labels=["watch"], threshold=threshold)

    boxes = [result["box"] for result in results]
    scores = [result["score"] for result in results]

    filtered_boxes = filter_overlapping_boxes(boxes, scores, iou_threshold=0.005)

    filtered_results = []
    for box in filtered_boxes:
        for result in results:
            if result["box"] == box:
                filtered_results.append(result)
                break

    return filtered_results


async def extractWithOpenAI(question, model="gpt-35-turbo"):
    messages = [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    client = AsyncAzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        api_version=os.getenv("OPENAI_API_VERSION", ""),
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT", ""),
    )

    chat_completion = await client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        messages=messages,
    )
    content = chat_completion.choices[0].message.content

    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        # 如果没有找到代码块中的 JSON，尝试直接匹配 JSON 对象
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        else:
            # 如果没有找到 JSON 对象，可以抛出异常或者返回一个默认值
            content = None

    return content


def filter_overlapping_boxes(boxes, scores, iou_threshold):
    filtered_boxes = []
    seen = defaultdict(bool)

    for i, box1 in enumerate(boxes):
        if seen[i]:
            continue

        filtered_boxes.append(box1)
        for j, box2 in enumerate(boxes[i + 1 :], start=i + 1):
            box1_tensor = torch.tensor(
                [[box1["xmin"], box1["ymin"], box1["xmax"], box1["ymax"]]]
            )
            box2_tensor = torch.tensor(
                [[box2["xmin"], box2["ymin"], box2["xmax"], box2["ymax"]]]
            )
            iou = box_iou(box1_tensor, box2_tensor)[0][0].item()
            if iou > iou_threshold:
                if scores[i] > scores[j]:
                    seen[j] = True
                else:
                    seen[i] = True

    return filtered_boxes


script = """
    function isListItem(element) {
        if (!element || !element.parentElement) {
            return false;
        }
        var siblings = element.parentElement.children;
        var elementClassList = element.className;
        for (var i = 0; i < siblings.length; i++) {
            if (siblings[i] !== element && siblings[i].tagName === element.tagName) {
                var siblingClassList = siblings[i].className;
                if (levenshteinSimilarity(elementClassList, siblingClassList) > 0.9) { // 调整阈值
                    return true;
                }
            }
        }
        return false;
    }

    function findListParent(element) {
        var parent = element.parentElement;
        while (parent && parent.tagName !== 'BODY') {
            if (isListItem(element)) {
                return parent;
            }
            element = parent;
            parent = parent.parentElement;
        }
        return null;
    }

    var ele = document.elementFromPoint(arguments[0], arguments[1]);
    var listParent = findListParent(ele);
    return listParent;

    """


def inject_levenshtein_similarity(driver):
    levenshtein_script = """
    function levenshteinDistance(a, b) {
        const matrix = [];

        for (let i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }

        for (let j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) === a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }

        return matrix[b.length][a.length];
    }

    window.levenshteinSimilarity = function(a, b) {
        const maxLen = Math.max(a.length, b.length);
        if (maxLen === 0) return 1.0;
        const distance = levenshteinDistance(a, b);
        return (maxLen - distance) / maxLen;
    }
    """
    driver.execute_script(levenshtein_script)
    is_available = driver.execute_script(
        "return typeof levenshteinSimilarity === 'function';"
    )
    if not is_available:
        raise Exception("levenshteinSimilarity function not available after injection.")


def get_html_list(watch_boxes, driver):
    if not watch_boxes or not driver:
        return None, None

    inject_levenshtein_similarity(driver)

    try:
        watches = [
            {"x": watch["box"]["xmin"], "y": watch["box"]["ymin"]}
            for watch in watch_boxes
        ]

        parent_elements = [
            driver.execute_script(script, watch["x"] + 5, watch["y"] + 5)
            for watch in watches
        ]

        parent_elements = [dom for dom in parent_elements if dom is not None]

        if not parent_elements:
            return None, None

        counter = Counter(parent_elements)
        most_common_parent = counter.most_common(1)[0][0]

        if not most_common_parent:
            return None, None

        most_common_class = driver.execute_script(
            """
            var parent = arguments[0];
            var threshold = 0.8;
            var children = parent.children;
            var classNames = [];
            
            for (var i = 0; i < children.length; i++) {
                classNames.push(children[i].className);
            }
            
            // 使用字典记录相似的类名组
            var classGroups = [];
            
            for (var i = 0; i < classNames.length; i++) {
                var foundGroup = false;
                for (var j = 0; j < classGroups.length; j++) {
                    if (levenshteinSimilarity(classNames[i], classGroups[j][0]) >= threshold) {
                        classGroups[j].push(classNames[i]);
                        foundGroup = true;
                        break;
                    }
                }
                if (!foundGroup) {
                    classGroups.push([classNames[i]]);
                }
            }
            
            // 找到包含最多元素的组
            var mostCommonGroup = classGroups.reduce(function(a, b) {
                return a.length > b.length ? a : b;
            });
            var mostCommonClass = mostCommonGroup[0];
            
            return mostCommonClass;
            """,
            most_common_parent,
        )

        print(f"most_common_class------------{most_common_class}")

        children_html_list = driver.execute_script(
            """
            var parent = arguments[0];
            var mostCommonClass = arguments[1];
            var threshold = 0.8;
            var children = parent.children;
            var htmlList = [];
            
            for (var i = 0; i < children.length; i++) {
                var className = children[i].className;
                var similarity = levenshteinSimilarity(className, mostCommonClass);
                if (similarity >= threshold) {
                    htmlList.push(children[i].outerHTML);
                }
            }
            return htmlList;
            """,
            most_common_parent,
            most_common_class,
        )

        print(f"Found {len(children_html_list)} DOM elements-----------------")

        return list(set(children_html_list)), most_common_parent.get_attribute(
            "class"
        ) or most_common_parent.get_attribute("id")
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def handle_popup(driver):
    # 定义一组可能的定位器
    locators = [
        # Text
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accept')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'agree')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'cookies')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'alle')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accetta')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'aceptar')]",
        ),
        # Id
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accept')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'agree')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'cookies')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'alle')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accetta')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'aceptar')]",
        ),
        # Class
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accept')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'agree')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'cookies')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'alle')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'accetta')]",
        ),
        (
            By.XPATH,
            "//*[contains(translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'aceptar')]",
        ),
    ]

    for locator in locators:
        try:
            # 尝试找到并点击按钮
            button = driver.find_element(*locator)
            button.click()
            return
        except Exception as e:
            # 如果当前定位器未找到按钮，继续尝试下一个定位器
            continue


def upload_html_to_s3(html_content):

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    bucket_name = os.getenv("S3_BUCKET_NAME", "")
    object_name = os.getenv("S3_BUCKET_HTML_SAVE_PATH", "")
    region_name = os.getenv("S3_BUCKET_REGION_NAME", "")

    print(f"aws_access_key_id: {aws_access_key_id}")
    print(f"aws_secret_access_key: {aws_secret_access_key}")
    print(f"bucket_name: {bucket_name}")
    print(f"object_name: {object_name}")

    if not html_content:
        print("Error: No HTML content provided.")
        return None

    if (
        not aws_access_key_id
        or not aws_secret_access_key
        or not bucket_name
        or not object_name
    ):
        print("Error: Missing environment variables.")
        return None

    # Correct way to create the S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    uuid_v4 = str(uuid.uuid4())

    object_name = f"{object_name}/{uuid_v4}"

    try:
        # upload HTML to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=html_content,
            ContentType="text/html",  # set the content type to HTML
        )
        print(f"Successfully uploaded to {bucket_name}/{object_name}")
        return uuid_v4
    except NoCredentialsError:
        print("Error: No AWS credentials found.")
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials found.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")

    return None


def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # 滚动到页面底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 等待内容加载

        # 再次获取新的页面高度
        new_height = driver.execute_script("return document.body.scrollHeight")

        # 检查是否高度变化，如果没有变化则表示已到达页面底部
        if new_height == last_height:
            break
        last_height = new_height


def is_list_item(element):
    if not element or not element.parent:
        return False
    siblings = element.parent.find_all(element.name)
    for sibling in siblings:
        if sibling != element:
            element_classes = set(element.get("class", []))
            sibling_classes = set(sibling.get("class", []))
            if element_classes & sibling_classes:
                return True
    return False


def find_list_parent(element):
    parent = element.parent
    while parent and parent.name != "body":
        if is_list_item(element):
            return parent
        element = parent
        parent = parent.parent
    return None


def get_api_html_list(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    all_elements = soup.find_all(True)

    parent_elements = []
    for element in all_elements:
        list_parent = find_list_parent(element)
        if list_parent:
            parent_elements.append(list_parent)

    # 如果没有找到任何 parent_elements，检查是否只有一个顶层元素
    if not parent_elements:
        parent_element = soup.find(True)
        if parent_element:
            children_html_list = [
                str(child) for child in parent_element.find_all(recursive=False)
            ]
            return children_html_list, ""  # 返回空字符串作为 parent
        else:
            return [], ""

    # 统计每个 parent 元素的引用
    counter = Counter(parent_elements)
    most_common_parent, _ = counter.most_common(1)[0]

    # 获取出现次数最多的 parent 元素的所有子元素的 outerHTML
    children_html_list = []
    for child in most_common_parent.find_all(recursive=False):
        children_html_list.append(str(child))

    print(f"Found {len(children_html_list)} DOM elements-----------------")

    return children_html_list, most_common_parent.get("class", "")


# 定义一个函数来将JSON数据转换为文本字符串
def _json_to_text(json_data):
    if isinstance(json_data, dict):
        return " ".join(
            [f"{key}: {_json_to_text(value)}" for key, value in json_data.items()]
        )
    elif isinstance(json_data, list):
        return " ".join([_json_to_text(item) for item in json_data])
    else:
        return str(json_data)


def _is_watch_related(text):
    if not text.strip():  # 检查文本是否为空
        return 0.0
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    model_path = os.path.join(current_dir, "bart-large-mnli")

    # 加载模型和处理器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    candidate_labels = ["watch", "not watch"]
    _classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    result = _classifier(text, candidate_labels)
    score = (
        result["scores"][0] if result["labels"][0] == "watch" else result["scores"][1]
    )
    watch_related_fields = ["price", "brand", "reference"]
    # 如果text中包含关键字段，则加1分
    for field in watch_related_fields:
        if field in text.lower():
            score += 1
            break
    return score


# 递归查找JSON中的所有数组
def _find_all_arrays(json_data):
    arrays = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            arrays.extend(_find_all_arrays(value))
    elif isinstance(json_data, list):
        if len(json_data) > 1:
            arrays.append(json_data)
        for item in json_data:
            arrays.extend(_find_all_arrays(item))
    return arrays


# 查找与手表相关性最高的数组
def find_most_related_array(json_data):
    arrays = _find_all_arrays(json_data)
    max_score = 0
    best_array = None
    for array in arrays:
        text = _json_to_text(array)
        score = _is_watch_related(text)
        if score > max_score:
            max_score = score
            best_array = array
    return best_array, max_score


def calculate_area(box):
    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]
    return width * height


def get_detail_images_html(watch_boxes, driver):
    inject_levenshtein_similarity(driver)
    if not watch_boxes:
        return None
    if not driver:
        return None

    try:
        # 找到包含最大区域的检测框
        max_detection = max(watch_boxes, key=lambda x: calculate_area(x["box"]))

        # 计算调整后的坐标
        device_pixel_ratio = driver.execute_script("return window.devicePixelRatio;")
        adjusted_x = (max_detection["box"]["xmin"] + 40) / device_pixel_ratio
        adjusted_y = (max_detection["box"]["ymin"] + 40) / device_pixel_ratio

        # 获取目标网页元素
        dom = driver.execute_script(
            """
            function isListItem(element) {
                if (!element || !element.parentElement) {
                    return false;
                }
                var siblings = element.parentElement.children;
                var elementClassList = Array.from(element.classList);
                for (var i = 0; i < siblings.length; i++) {
                    if (siblings[i] !== element && siblings[i].tagName === element.tagName) {
                        var siblingClassList = Array.from(siblings[i].classList);
                        if (elementClassList.some(cls => siblingClassList.includes(cls))) {
                            return true;
                        }
                    }
                }
                return false;
            }

            function findListParent(element) {
                var parent = element.parentElement;
                while (parent && parent.tagName !== 'BODY') {
                    if (isListItem(element)) {
                        return parent;
                    }
                    element = parent;
                    parent = parent.parentElement;
                }
                return null;
            }

            var ele = document.elementFromPoint(arguments[0], arguments[1]);
            var listParent = findListParent(ele);
            return listParent;

            """,
            adjusted_x,
            adjusted_y,
        )

        # 获取该元素的外部 HTML
        html = driver.execute_script("return arguments[0].outerHTML;", dom)

        return html

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
