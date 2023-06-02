from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)

cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

camera = cv2.VideoCapture(0)
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            # Display the resulting frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
