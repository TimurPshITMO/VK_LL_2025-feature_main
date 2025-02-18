from typing import List
from pydantic import BaseModel

class Request(BaseModel):
    cpm: int
    hour_start: int
    hour_end: int
    publishers: str
    audience_size: int
    user_ids: str


class Response(BaseModel):
    at_least_one: float
    at_least_two: float
    at_least_three: float
    
 