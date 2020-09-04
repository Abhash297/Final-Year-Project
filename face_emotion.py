from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from playsound import playsound
import time
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default), "rgb", or "lab"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# starting video streaming
cv2.namedWindow('my_face')
camera = cv2.VideoCapture(0)
time.sleep(2)

color = args['color']
bins = args['bins']
resizeWidth = args['width']

# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
elif color == 'lab':
    ax.set_title('Histogram (L*a*b*)')
else:
    ax.set_title('Histogram (grayscale)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
elif color == 'lab':
    lineL, = ax.plot(np.arange(bins), np.zeros((bins,)), c='k', lw=lw, alpha=alpha, label='L*')
    lineA, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='a*')
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='y', lw=lw, alpha=alpha, label='b*')
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
ax.legend()
plt.ion()
plt.show()



while True:
    frame = camera.read()[1]
    #reading the frame



    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
            interpolation=cv2.INTER_AREA)
    # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(frame.shape[:2])
    if color == 'rgb':
        cv2.imshow('RGB', frame)
        (b, g, r) = cv2.split(frame)
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
    elif color == 'lab':
        cv2.imshow('L*a*b*', frame)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        (l, a, b) = cv2.split(lab)
        histogramL = cv2.calcHist([l], [0], None, [bins], [0, 255]) / numPixels
        histogramA = cv2.calcHist([a], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineL.set_ydata(histogramL)
        lineA.set_ydata(histogramA)
        lineB.set_ydata(histogramB)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray)
        histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
        lineGray.set_ydata(histogram)
    fig.canvas.draw()
####        



    
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((600, 700, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification 
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
    else:continue
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)


                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 255, 0), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 255, 0), 2)



    cv2.imshow('my_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
