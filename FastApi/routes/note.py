from fastapi import APIRouter
from models.note import Note
from config.db import conn
from schemas.note import noteEntity, notesEntity
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


note=APIRouter()
templates = Jinja2Templates(directory="templates")



@note.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    docs=conn.notes.notes.find({})
    newdoc=[]
    for doc in docs:
        newdoc.append({
            "id":doc["_id"],
            "title":doc["title"],
            "desc":doc["desc"],
            "note":doc["note"],
            "important":doc["important"]
        })
    return templates.TemplateResponse(name="index.html", context={"request": request, "newdoc":newdoc})



@note.post("/")
async def create_item(request: Request):
    form = await request.form()
    note_data = {
        "title": form.get("title"),
        "desc": form.get("desc"), 
        "note": form.get("note"),
        "important": form.get("important") == "on"
    }
    conn.notes.notes.insert_one(note_data)
    # Redirect to GET route to show updated list
    docs = conn.notes.notes.find({})
    newdoc = []
    for doc in docs:
        newdoc.append({
            "id": doc["_id"],
            "title": doc["title"],
            "desc": doc["desc"],
            "note": doc["note"],
            "important": doc["important"]
        })
    return templates.TemplateResponse(name="index.html", context={"request": request, "newdoc": newdoc})
