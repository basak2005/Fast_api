from pydantic import BaseModel


class Note(BaseModel):
    title: str
    desc: str
    note: str  # Added missing note field
    important: bool = False