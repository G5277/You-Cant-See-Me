from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import uvicorn
import io
from PIL import Image
import base64
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global background frame variable
bg_frame = None

# HSV Ranges for different cloak colors
HSV_RANGES = {
    "red": ([0, 120, 70], [10, 255, 255]),  # Red
    "blue": ([90, 100, 50], [130, 255, 255]),  # Blue
    "green": ([40, 40, 40], [80, 255, 255]),  # Green
}

@app.post("/capture-background/")
async def capture_background(file: UploadFile = File(...)):
    """ Capture and store the initial background frame """
    global bg_frame
    image = Image.open(io.BytesIO(await file.read()))
    bg_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print("Background frame captured!")  # ✅ Debugging
    return {"message": "Background captured successfully"}

@app.post("/process-frame/")
async def process_frame(file: UploadFile = File(...), cloak_color: str = Form(...)):
    """ Process the current frame and apply invisibility effect """
    global bg_frame
    if bg_frame is None:
        return JSONResponse(content={"error": "Background frame not set"}, status_code=400)

    # Read input image
    image = Image.open(io.BytesIO(await file.read()))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Validate cloak color
    if cloak_color not in HSV_RANGES:
        return JSONResponse(content={"error": "Invalid cloak color"}, status_code=400)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv, upper_hsv = np.array(HSV_RANGES[cloak_color][0], dtype=np.uint8), np.array(HSV_RANGES[cloak_color][1], dtype=np.uint8)

    # Apply mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)

    mask_inv = cv2.bitwise_not(mask)

    # Extract the background and the foreground
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    inv_area = cv2.bitwise_and(bg_frame, bg_frame, mask=mask)

    # Blend images
    final = cv2.addWeighted(bg, 1, inv_area, 1, 0)

    # Encode image as base64
    _, buffer = cv2.imencode(".jpg", final)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"image": image_base64})
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)