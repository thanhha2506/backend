from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from app.routers import predict
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="FruitVision API")

static_path = os.path.join(os.getcwd(), "app", "static")
os.makedirs(static_path, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_path), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "Welcome to FruitVision API"}
