def noteEntity(item)->dict:
    return {
        "_id":str(item["_id"]),
        "title":item["title"],
        "desc":item["desc"],
        "note":item["note"],  # Added missing note field
        "important":item["important"]
    }



def notesEntity(items)->list:
    return [noteEntity(item) for item in items]