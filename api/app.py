from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .utils import OPENAI_API_KEY

app = FastAPI(
    title="Sermon Search API",
    description="API for searching sermon transcripts and generating answers from the content",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fellowship-digital-ministry.github.io", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
