from fastapi import FastAPI, UploadFile, File
from pipeline import get_embedding

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
async def embed(image: UploadFile = File(...)):
    embedding = await get_embedding(image)
    return {"embedding": embedding}