from fastapi import FastAPI, UploadFile, File, HTTPException
from pipeline import get_embedding, ModelLoadError

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
async def embed(image: UploadFile = File(...)):
    try:
        embedding = await get_embedding(image)
        return {"embedding": embedding}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ModelLoadError as e:
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from e
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error") from None