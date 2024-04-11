from tet import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints
import shutil
from make_folders import actions
from keras.models import load_model
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
import numpy as np
import cv2
from PIL import Image, ImageTk
from itertools import count
import string
import os

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# returns a compiled model
# identical to the previous one
model = load_model('keras_lstm.h5')


class TextInput(BaseModel):
    text: str


@app.post("/predict_text")
async def predict_text(file: UploadFile = File(...)):
    sequence = []
    sentence = []
    threshold = 0.8
    print(file.filename)
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
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-60:]

            if len(sequence) == 60:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

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

    return {'predicted_text': ' '.join(sentence)}


@app.post("/convert_text_to_video")
async def convert_text_to_video(text_input: TextInput):
    r = sr.Recognizer()
    isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
               'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
               'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
               'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
               'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing',
               'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
               'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
               'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
               'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
               'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
               'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
               'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
               'where is the bathroom', 'where is the police station', 'you are wrong', 'address', 'agra', 'ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
               'bihar', 'bihar', 'bridge', 'cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile', 'dasara',
               'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
               'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
               'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
               'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
               'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
               'voice', 'wednesday', 'weight', 'please wait for sometime', 'what is your mobile number', 'what are you doing', 'are you busy']
    arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    output_frames = []
    type_of = ""
    test = ''
    sentence = ''

    for word in text_input.text.split():
        word = word.lower()
        for c in string.punctuation:
            word = word.replace(c, "")
        sentence += word

    if text_input.text in isl_gif:
        print(word)
        test = 'y'
        type_of = "in_gif"
        gif_path = f"ISL_Gifs/{text_input.text}.gif"
        gif = cv2.VideoCapture(gif_path)
        while True:
            ret, frame = gif.read()
            if not ret:
                break
            output_frames.append(frame)
        gif.release()
    else:
        for c in word:
            if c in arr:
                type_of = "letter"
                image_path = f"letters/{c}.jpg"
                ImageItself = Image.open(image_path)
                ImageNumpyFormat = np.asarray(ImageItself)
                output_frames.append(ImageNumpyFormat)

    # Combine frames into video
    height, width, _ = output_frames[0].shape
    video_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))
    for frame in output_frames:
        video.write(frame)
    video.release()

    # Return video file
    return {"file_path": os.path.abspath(video_path), "type": type_of, 'test': test}
