import os
import time
import aioboto3
import uuid
import gc
import glob
import shutil
import re
import logging
import torch

from bs4 import BeautifulSoup
from tempfile import mkdtemp
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from seleniumwire import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
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
from fake_useragent import UserAgent


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
    agent = UserAgent().random
    logging.info(f"user-agent={agent}")
    options.add_argument(f"--user-agent={agent}")
    # options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    # PROXY_USERNAME = os.getenv("PROXY_USERNAME", None)
    # PROXY_PASS = os.getenv("PROXY_PASS", None)
    # PROXY_ENDPOINT = os.getenv("PROXY_ENDPOINT", None)
    # PROXY_PORT = os.getenv("PROXY_PORT", None)
    # logging.info(f"proxy_user_name: {PROXY_USERNAME}")
    # logging.info(f"proxy_pass: {PROXY_PASS}")
    # logging.info(f"proxy_endpoint: {PROXY_ENDPOINT}")
    # logging.info(f"proxy_port: {PROXY_PORT}")
    # options.add_argument(f'--proxy-server=http://127.0.0.1:7890')
    # proxies_extension = proxies(PROXY_USERNAME, PROXY_PASS, PROXY_ENDPOINT, PROXY_PORT)
    # options.add_extension(proxies_extension)

    seleniumwire_options = {}
    # use proxy if PROXY_URL
    PROXY_URL = os.getenv("PROXY", None)
    if PROXY_URL:
        seleniumwire_options = {
            "proxy": {"http": f"{PROXY_URL}", "https": f"{PROXY_URL}"},
        }
        logging.info(f"use proxy {PROXY_URL}")
    else:
        logging.info("no proxy")

    service = ChromeService(ChromeDriverManager().install())
    try:
        # 添加延迟确保 Chrome 完全启动

        driver = webdriver.Chrome(
            service=service,
            seleniumwire_options=seleniumwire_options,
            options=options,
        )
        driver.pending_requests_count = 0
        driver.request_interceptor = lambda request: request_interceptor(
            request, driver
        )
        driver.response_interceptor = lambda request, response: response_interceptor(
            request, response, driver
        )

        driver.set_page_load_timeout(600)
    except Exception as e:
        logging.info(f"Error starting Chrome WebDriver: {e}")
        raise

    # 返回临时目录路径和 driver
    return driver, [user_data_dir, data_path, disk_cache_dir, homedir]


# 清理临时目录和关闭 WebDriver 的方法
def clean_up_driver(driver, temp_dirs):
    try:
        driver.quit()
        logging.info("Chrome WebDriver successfully quit.")
    except Exception as e:
        logging.info(f"Error quitting Chrome WebDriver: {e}")

    for dir_path in temp_dirs:
        try:
            shutil.rmtree(dir_path)
            logging.info(f"Temporary directory {dir_path} successfully removed.")
        except Exception as e:
            logging.info(f"Error removing temporary directory {dir_path}: {e}")
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
            logging.info(f"Deleted file: {file}")


def get_tmp_files():
    for root, dirs, files in os.walk("/tmp"):
        for file in files:
            logging.info(f"clean_up_driver {os.path.join(root, file)}")


def check_space(title):
    total, used, free = shutil.disk_usage("/tmp")
    logging.info(
        f"{title} Total: {total/1024/1024} MB, Used: {used/1024/1024} MB, Free: {free/1024/1024} MB"
    )


# 启用CDP监听
def enable_request_logging(driver):
    driver.execute_cdp_cmd("Network.enable", {})


def is_ajax_request(request):
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.headers.get("Content-Type")
        and "application/json" in request.headers.get("Content-Type")
    )


def request_interceptor(request, driver):
    if is_ajax_request(request):
        # 使用 driver 实例的 pending_requests_count 属性
        driver.pending_requests_count += 1


def response_interceptor(request, response, driver):
    if is_ajax_request(request):
        # 使用 driver 实例的 pending_requests_count 属性
        driver.pending_requests_count -= 1


def wait_for_requests_to_complete(driver, timeout=30):
    try:
        start_time = time.time()

        WebDriverWait(driver, timeout).until(lambda d: d.pending_requests_count == 0)

        total_time = time.time() - start_time
        logging.info(f"Waited for XHR requests to complete: {total_time:.2f} seconds")
    except TimeoutException:
        logging.info("Timeout waiting for requests, but continuing execution.")


# 使用 WebDriverWait 等待图片完全加载
def wait_for_images_to_load(driver, timeout=30):
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script(
                """
                let images = Array.from(document.images);
                return images.every(img => img.complete && img.naturalWidth > 0);
                """
            )
        )
        logging.info("Images loaded successfully.")
    except TimeoutException:
        logging.info("Timeout waiting for images to load, but continuing execution.")


def watch_detect(image, threshold=0.001):
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
    openAiClient = AsyncAzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        api_version=os.getenv("OPENAI_API_VERSION", ""),
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT", ""),
    )
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

    chat_completion = await openAiClient.chat.completions.create(
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
                // 过滤掉没有有效 outerHTML 和内容为空的元素
                if (children[i].outerHTML && children[i].outerHTML.trim() !== '' && children[i].textContent.trim() !== '') {
                    console.log('item class name: ' + children[i].className);
                    classNames.push(children[i].className);
                }
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

        logging.info(f"most_common_class------------{most_common_class}")

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

        return list(set(children_html_list)), most_common_parent.get_attribute(
            "class"
        ) or most_common_parent.get_attribute("id")
    except Exception as e:
        logging.info(f"Error: {e}")
        return None, None


def handle_popup(driver):
    # JavaScript代码来查找并点击与Cookies相关的按钮，首先检查普通元素，再检查shadowRoot
    script = """
    function closeCookiePopups() {
        const keywords = ['accept', 'agree', 'cookie', 'alle', 'accetta', 'alla', 'aceptar', 'continue'];
        let foundElements = [];

        // 定义一个函数来查找和点击按钮
        function findAndClickButtons(root) {
            keywords.forEach(keyword => {
                const lowerKeyword = keyword.toLowerCase();
                const buttons = root.querySelectorAll('button, input[type="button"], input[type="submit"], div[role="button"], span[role="button"], a');

                buttons.forEach(button => {
                    const buttonText = button.textContent.toLowerCase().trim();
                    
                    // 使用正则表达式来匹配
                    const regex = new RegExp('\\\\b' + lowerKeyword + '\\\\b', 'i');

                    // 获取按钮的样式
                    const style = window.getComputedStyle(button);

                    // 检查按钮是否在可视区域内
                    const rect = button.getBoundingClientRect();
                    const isVisible = rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth);

                    // 检查display和visibility属性，以及按钮是否在可见区域
                    const isDisplayed = style.display !== 'none' && style.visibility !== 'hidden' && isVisible;

                    // 检查href属性是否会跳转
                    const href = button.getAttribute('href');  // 获取href属性的值
                    const isValidHref = href == null || href == "" || (!href.startsWith("http") && !href.startsWith("/"));

                    if (isDisplayed &&
                        isValidHref && 
                        (regex.test(buttonText) || 
                        (button.value && regex.test(button.value.toLowerCase().trim())))) {
                        
                        foundElements.push(button.outerHTML);
                        button.click();
                    }
                });
            });
        }

        // 首先查找普通元素中的按钮
        findAndClickButtons(document);

        // 查找包含 shadow-root 的元素
        const shadowHostElements = document.querySelectorAll('body > *');

        shadowHostElements.forEach(element => {
            // 尝试获取 shadow-root
            const shadowRoot = element.shadowRoot;
            if (shadowRoot) {
                // 查找 shadow-root 中的按钮
                findAndClickButtons(shadowRoot);
            }
        });

        return foundElements;
    }

    // 执行关闭操作并返回找到的元素信息
    return closeCookiePopups();
    """

    # 执行 JavaScript 脚本，并返回找到的元素信息
    return driver.execute_script(script)


session = aioboto3.Session()


async def upload_html_to_s3(html_content):
    # 从环境变量中获取 AWS 相关配置
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    bucket_name = os.getenv("S3_BUCKET_NAME", "")
    object_name = os.getenv("S3_BUCKET_HTML_SAVE_PATH", "")
    region_name = os.getenv("S3_BUCKET_REGION_NAME", "")

    # 检查 HTML 内容是否存在
    if not html_content:
        logging.info("Error: No HTML content provided.")
        return None

    # 检查是否所有必要的环境变量都有值
    if not all([aws_access_key_id, aws_secret_access_key, bucket_name, object_name]):
        logging.info("Error: Missing environment variables.")
        return None

    # 生成唯一的文件名
    uuid_v4 = str(uuid.uuid4())
    object_name = f"{object_name}/{uuid_v4}"

    # 使用 aioboto3 进行异步 S3 客户端的操作
    async with session.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    ) as s3:
        try:
            # 异步上传 HTML 内容到 S3
            await s3.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=html_content,
                ContentType="text/html",  # 设置内容类型为 HTML
            )
            logging.info(f"Successfully uploaded to {bucket_name}/{object_name}")
            return uuid_v4
        except NoCredentialsError:
            logging.info("Error: No AWS credentials found.")
        except PartialCredentialsError:
            logging.info("Error: Incomplete AWS credentials found.")
        except Exception as e:
            logging.info(f"Error uploading to S3: {e}")
            return None


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

    logging.info(f"Found {len(children_html_list)} DOM elements-----------------")

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
        logging.info(f"Error occurred: {e}")
        return None
