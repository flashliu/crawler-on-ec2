from typing import Optional
from pydantic import BaseModel


class ScrapListBrowserInfo(BaseModel):
    url: str
    parent: Optional[str] = None
