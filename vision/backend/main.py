from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import asyncio


app = FastAPI()

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")



########   Test with Repliacte Model ########
import replicate

async def call_llava(img, prompt):
    output = await replicate.run(
        "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
        input={
            "image": img,
            "top_p": 1,
            "prompt": "Provide a detailed description of what you see in the picture and also " + prompt,
            "history": [],
            "max_tokens": 1024,
            "temperature": 0.2
        }
    )

    # Assuming 'output' is a string or can be converted to a string
    response = str(output)
    return io.BytesIO(response.encode())

app = FastAPI()

@app.post("/llm-on-image")
async def classify_image(file: UploadFile = File(...), prompt: str = ""):
    contents = await file.read()
    img_np = np.array(bytearray(contents), dtype=np.uint8)
    response = await call_llava(img_np, prompt)

    return StreamingResponse(response, media_type="text/plain")

@app.websocket_route("/live_feed")
async def live_feed(websocket: WebSocket):
    await websocket.accept()

    try:
        for _ in range(15):  # Adjust the duration as needed
            # Capture a frame from a live video feed (replace with your source)
            frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)  # Placeholder for live feed

            # Process the frame
            processed_frame = replicate_model(frame)

            # Send the processed frame to the client
            await websocket.send_bytes(cv2.imencode('.jpg', processed_frame)[1].tobytes())
            await asyncio.sleep(1)  # Adjust the delay as needed for your use case

    except asyncio.CancelledError:
        pass
    finally:
        await websocket.close()

app.mount("/", StaticFiles(directory="static", html=True), name="static")