import cv2,numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
from tet import mp_holistic,mediapipe_detection,draw_styled_landmarks,extract_keypoints
from keras.models import load_model
from make_folders import actions
import os
    # returns a compiled model
    # identical to the previous one

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model('keras_lstm.h5')
import shutil
@app.post("/predict_text")
async def predict_text(file: UploadFile = File(...)):
    sequence = []
    sentence = []
    threshold = 0.8
    # Read the video file
    try:
         with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        return {"error": str(e)}


    cap = cv2.VideoCapture(file.filename)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while True:

            # Read feed
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            fianl_sen=[]
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]
            
            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                fianl_sen.append(actions[np.argmax(res)])
                
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                    
            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
            
        cap.release()
        os.remove(file.filename)
        

    return {'predicted_text': ' '.join(sentence),
            "sent":fianl_sen}

