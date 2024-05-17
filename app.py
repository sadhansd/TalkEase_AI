from flask import Flask, render_template,request,redirect, url_for,Response
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
import pickle
import mediapipe as mp
import base64


app = Flask(__name__, static_folder='static')
imgfold = os.path.join('static', 'images')

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/instruction')
def instructions():
    return render_template('instruction.html')

@app.route('/select_user')
def user_view():
    return render_template('users.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        opt1 = request.form['option1']
        opt2 = request.form['option2']

        if(opt1 == opt2):
            alert_message = "Please select different options"
            return render_template('users.html', alert_message=alert_message)

        if( opt1 == '1'):
            if(opt2 == '2'):
                imgUrl = "static/images/speechtotexttitle.png"
                return render_template('text_to_speak.html',imgUrl=imgUrl)
            if(opt2 == '3'):
                imgUrl = "static/images/speechtosigntitle.png"
                return render_template('speech_to_sign.html', imgUrl=imgUrl)
        if(opt1 == '2'):
            if(opt2 == '1'):
                imgUrl = "static/images/texttospeechtitle.png"
                return render_template('text_to_speak.html',imgUrl=imgUrl)
            if(opt2 == '3'):
                imgUrl = "static/images/texttosigntitle.png"
                return render_template('text_to_sign.html', imgUrl=imgUrl)
        if(opt1 == '3'):
            if(opt2 == '1'):
                text = "dumb to blind - sign to speech"
                imgUrl = "static/images/signtospeechtitle.png"
                return render_template('speech_to_sign.html', imgUrl=imgUrl)
            if(opt2 == '2'):
                text = "dumb to deaf - sign to text"
                imgUrl = "static/images/signtotexttitle.png"
                return render_template('text_to_sign.html', imgUrl=imgUrl)

# ------------DEAF AND BLIND------------

recognizer = sr.Recognizer()

@app.route('/text_to_speak', methods=['POST'])
def text_to_speak():
    if request.method == 'POST':
        text_to_convert = request.form['text']

        tts = gTTS(text=text_to_convert, lang='en')  # 'en' stands for English

        # Save the converted audio to a file (you can change the file format if needed, e.g., 'output.mp3')
        output_file = "output.mp3"
        tts.save(output_file)

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        # Convert and play the text as speech using pyttsx3
        engine.say(text_to_convert)
        engine.runAndWait()

        return render_template('text_to_speak.html')

@app.route('/speech_to_text', methods=['GET', 'POST'])
def speech_to_text():
    if request.method == 'POST':

        output_text = "Audio"

        with sr.Microphone() as source:
            imgUrl = os.path.join(imgfold, 'speaker.png')
            audio = recognizer.listen(source)
        try:
            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your audio.")
        except sr.RequestError as e:
            print(f"Sorry, there was an error connecting to the Google API: {e}")

        # Define a dictionary of common text abbreviations and their expansions
        abbreviations = {
            "r": "are",
            "u": "you",
        }

        # Function to replace abbreviations with their expansions
        def replace_abbreviations(text, abbreviation_dict):
            words = text.split()
            expanded_words = [abbreviation_dict.get(word, word) for word in words]
            expanded_text = " ".join(expanded_words)
            return expanded_text

        # Convert the input text
        output_text = replace_abbreviations(text, abbreviations)

        if (output_text == "hi" or output_text == "hello"):
            output_text = "hi"
    return render_template('text_to_speak.html', text=output_text)

# -----------DEAF AND DUMB-----------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']


labels = data_dict['labels']


connections = data_dict['connection']

dataset = ["hello","i love you","yes","no","back","please","doubt","right","left","bye"]

def drawlines(val):
    image = np.zeros((600, 600, 3), dtype=np.uint8) * 255
    index = labels.index(val)

    landmark_points = data[index]

    landmark_points = [[landmark_points[i], landmark_points[i + 1]] for i in range(0, len(landmark_points), 2)]

    connection_points = connections[index]
    connection_points = [[connection_points[i], connection_points[i + 1]] for i in
                         range(0, len(connection_points), 2)]

    points = []

    for landmark in landmark_points:
        x, y = int(landmark[0] * 600), int(landmark[1] * 600)
        points.append((x, y))

    for point in points:
        cv2.circle(image, point, 5, (0, 0, 255), -1)

    for connection in connections:
        start_point = points[connection[0]]
        end_point = points[connection[1]]
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    return image

@app.route('/text_to_sign',methods=['GET','POST'])
def text_to_sign():
    if request.method == 'POST':
        text = request.form['text']
        text = text.lower()
        val = str(dataset.index(text))
        image = drawlines(val)

        img_file = "./static/sign_images/" + text + ".png"
        cv2.imwrite(img_file, image)

    return render_template('text_to_sign.html', image=img_file)


@app.route('/speech_to_sign', methods=['GET', 'POST'])
def speech_to_sign():
    if request.method == 'POST':

        output_text = "Audio"

        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        try:
           
            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your audio.")
        except sr.RequestError as e:
            print(f"Sorry, there was an error connecting to the Google API: {e}")

        # Define a dictionary of common text abbreviations and their expansions
        abbreviations = {
            "r": "are",
            "u": "you",
        }

        # Function to replace abbreviations with their expansions
        def replace_abbreviations(text, abbreviation_dict):
            words = text.split()
            expanded_words = [abbreviation_dict.get(word, word) for word in words]
            expanded_text = " ".join(expanded_words)
            return expanded_text

        # Convert the input text
        output_text = replace_abbreviations(text, abbreviations)

        

        output_text = output_text.lower()
        val = str(dataset.index(output_text))
        image = drawlines(val)

        img_file = "./static/sign_images/" + text + ".png"
        cv2.imwrite(img_file, image)

    return render_template('speech_to_sign.html', image=img_file, output=output_text)



# webcam = None
#
# def generate_frames():
#     global webcam
#     while True:
#         if webcam is not None:
#             success, frame = webcam.read()
#             if not success:
#                 break
#             # Encode the frame to JPEG format
#
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 break
#             # Convert the frame buffer to bytes
#             frame_bytes = buffer.tobytes()
#             # Yield the frame bytes to the Response generator
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# @app.route('/signDet')
# def detection():
#     return render_template('text_to_sign.html', video_feed_url="{{ url_for('video_feed') }}")

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variable to manage webcam status

model = load_model('action.h5')

colors = [(245,117,16), (117,245,16), (16,117,245), (135,206,235), (255,192,203)]
webcam = None
speech = None
actions = np.array(['hello', 'thanks', 'help', 'yes', 'home'])
sequence = []
sentence = []
predictions = []
threshold = 0.5

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def predicted_speech(sentence):
    for text in sentence:
        tts = gTTS(text=text, lang='en')  # 'en' stands for English
        output_file = "output.mp3"
        tts.save(output_file)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()


def generate_frames():
    global webcam
    actions = np.array(['hello', 'thanks', 'help','home','yes'])
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    while True:
        if webcam is not None:
            success, frame = webcam.read()
            if not success:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                text = actions[np.argmax(res)]
                print(text)

                predictions.append(np.argmax(res))
                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
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


            cv2.rectangle(image, (0, 420), (640, 500), (0, 0, 0), -1)
            cv2.putText(image, ' '.join(sentence), (3, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if speech is not None:
                predicted_speech(sentence)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                break

            # Convert the frame buffer to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame bytes to the Response generator
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_webcam')
def toggle_webcam():
    global webcam

    if webcam is None:
        webcam = cv2.VideoCapture(0)
    else:
        webcam.release()
        webcam = None
    return 'Success'

@app.route('/toggle_webcam_speech')
def toggle_webcam_speech():
    global webcam
    global speech

    if webcam is None:
        webcam = cv2.VideoCapture(0)
        speech = not None
    else:
        webcam.release()
        webcam = None
    return 'Success'

if __name__  == '__main__':
    app.run(debug=True)