from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Tea(BaseModel):
    id: Optional[int] = None
    name: str
    type: str
    origin: str

# In-memory storage for teas
teas: List[Tea] = []

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tea API"}

@app.get("/teas")
def get_all_teas():
    return teas

@app.get("/teas/{tea_id}")
def get_tea(tea_id: int):
    for tea in teas:
        if tea.id == tea_id:
            return tea
    raise HTTPException(status_code=404, detail="Tea not found")

@app.post("/teas")
def create_tea(tea: Tea):
    teas.append(tea)
    return tea



@app.put("/teas/{tea_id}")
def update_tea(tea_id: int, updated_tea: Tea):
    for i, tea in enumerate(teas):
        if tea.id == tea_id:
            updated_tea.id = tea_id
            teas[i] = updated_tea
            return updated_tea
    raise HTTPException(status_code=404, detail="Tea not found")

@app.delete("/teas/{tea_id}")
def delete_tea(tea_id: int):
    for i, tea in enumerate(teas):
        if tea.id == tea_id:
            del teas[i]
            return {"message": "Tea deleted successfully"}
    raise HTTPException(status_code=404, detail="Tea not found")
