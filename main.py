
from libs.neural_network import *
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from io import BytesIO

import uvicorn


app = FastAPI()
nn  = load_nn("model/model.npy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
  return "go to /predict or /predict_str and make a post request with files"

@app.post("/predict_str")
async def predict_str(request: Request):
  try:
    data = await request.json()

    prediction = nn.forward([float(pixel) for pixel in data["img_str"].split(",")])

    return {"cat": f"{int(prediction[0] * 100)}%", "rabbit": f"{int(prediction[1] * 100)}%"}
  except Exception as e:
    return str(e)
  

@app.post("/predict")
async def predict(file : UploadFile = File(...)):
  try:
    img_from_bytes : Image = ImageOps.invert(Image.open(BytesIO(await file.read())).convert("L"))
    img : np.ndarray = np.array(img_from_bytes.resize((28, 28),Image.ANTIALIAS)).flatten()

    for i in range(len(img)):
      img[i] = (255 if img[i] > 0 else 0) / 255
    
    prediction = nn.forward(img)

    return {"cat": f"{int(prediction[0] * 100)}%", "rabbit": f"{int(prediction[1] * 100)}%"}
  except Exception as e:
    return str(e)

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)