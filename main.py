import warnings
from dotenv import load_dotenv
from fastapi import FastAPI
from routes.scrap_list_browser import router as scrap_list_browser_router
from routes.scrap_list_html import router as scrap_list_html_router
from routes.scrap_list_json import router as scrap_list_json_router
from routes.scrap_detail import router as scrap_detail_router

load_dotenv(override=True)

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*TLS in TLS.*")

app = FastAPI()

app.include_router(scrap_list_browser_router)
app.include_router(scrap_list_html_router)
app.include_router(scrap_list_json_router)
app.include_router(scrap_detail_router)
