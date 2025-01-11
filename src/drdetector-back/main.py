import os.path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import drdetector.load_chkpts
from drdetector import load_chkpts
from drdetector.config import *
from drdetector.cnn import *
from torchvision import transforms
from PIL import Image
from io import BytesIO
import logging

class_map = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Init FastAPI
app = FastAPI()

TEST_MODEL_CHECKPOINT = "models/cnn_alexnet_freeze_backbone_False.pth"


# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {"message": "Welcome to DRDetector!"}

@app.post("/predict")
async def predict(file: UploadFile = File()):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    filename = file.filename
    content_type = file.content_type

    # Read the file content
    file_content = await file.read()

    logging.info(f"Converting image {filename} to PIL.Image (RGB)")
    image = Image.open(BytesIO(file_content)).convert("RGB")

    logging.info(f"Converting image {filename} to tensor")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add dimension as the first axis (1, 3, 224, 224)
    logging.info(f"Image tensor shape {image_tensor.shape}")

    logging.info(f"Loading model")
    model = Classifier(len(CLASS_NAMES), backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE)
    model, _, _ = load_chkpts.load_checkpoint(model, TEST_MODEL_CHECKPOINT)
    print(f"\n===============> model, Type: {type(model)}")

    logging.info(f"Predicting image class")
    output = model(image_tensor)
    max_class = torch.argmax(output, dim=1).item()
    print("==============> Class with maximum value:", max_class)

    logging.info(f"Output shape {output.shape}, Output {output}")

    print("returning", JSONResponse({
        "filename": filename,
        "content_type": content_type,
        "file_size": len(file_content)
    }))

    return JSONResponse({
        "filename": filename,
        "content_type": content_type,
        "file_size": len(file_content),
        "image_class": class_map[max_class],
    })


