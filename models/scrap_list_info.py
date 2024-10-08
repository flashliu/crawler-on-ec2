from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


class ScrapListInfo(BaseModel):
    url: str
    detail_url_template: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    payload_type: Optional[Literal["json", "form"]] = "json"
    response_key:Optional[str] = None
