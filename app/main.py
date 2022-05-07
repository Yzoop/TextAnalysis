from typing import Optional

from fastapi import FastAPI
from nltk.tokenize import word_tokenize

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/analyze")
def analyze(text: str):
    text_size = len(text)
    num_tokens = len(word_tokenize(text))
    return {"textSize": text_size,
            "numWords": num_tokens}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}