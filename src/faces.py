from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from PIL import Image
import time
import io
import tflite_runtime.interpreter as tflite
import re

import numpy as np
import cv2
import pickle

from utils import CFEVideoConf, image_resize

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    # print(classes)
    # print(scores)
    for i in range(count):
        if scores[i] >= threshold: # i>0 and 
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def visualize_objects(img, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        startpoint = (xmin, ymin)
        end_point = (xmax, ymax)

        textlabel = '%s :  %.2f' % (labels[obj['class_id']], obj['score']*100) + '%'
        if "book" in textlabel:
            cv2.rectangle(img, startpoint, end_point ,color=(52, 235, 140), thickness=3) # Draw Rectangle with the coordinates
            #annotator.bounding_box([xmin, ymin, xmax, ymax])
            
            # print(obj)
            # print(int(obj['class_id']))
            # print(textlabel)
            text_size = 1
            cv2.putText(img, "ID-Card" , startpoint,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (52, 235, 140),thickness=2)
            #annotator.text([xmin, ymin], '%s\n%.2f' % (labels[obj['class_id']], obj['score']))

        if "cell" in textlabel and obj['score'] > 0.6:
            # print("cell -->")
            # print(obj['score'])
            # org
            org = (30, 100)
            # fontScale
            fontScale = 0.6
            # Blue color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'ALERT! Cellphone detected!',
                        org, font, fontScale, color, thickness, cv2.LINE_AA)


#####=====Main==============================================
            
def main():

    # face recognizer
    face_cascade = cv2.CascadeClassifier(
        'cascades/data/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./recognizers/face-trainner.yml")

    name_labels = {"person_name": 1}
    with open("pickles/face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        name_labels = {v: k for k, v in og_labels.items()}
    #End of Face recog


    # Object detect
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--image',
        default='data/00871.jpg',
        help='image to be classified')
    parser.add_argument(
        '--model', default='coco_ssd_mobilenet_v1_1/detect.tflite', help='File path of .tflite file.')
    parser.add_argument(
        '--labels', default='coco_ssd_mobilenet_v1_1/labelmap.txt', help='File path of labels file.')#, required=True
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    # print(labels[61])
    # print(labels[83])

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(args.model)
    interpreter.allocate_tensors()

    # Get input tensor details
    input_details = interpreter.get_input_details()
    # print(input_details)
    output_details = interpreter.get_output_details()
    # print(output_details)

    # check the type of the input tensor
    #input_details[0]['dtype']
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    # print(height)
    width = input_details[0]['shape'][2]
    # print(width)
    # End of obj detect


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open video device")
     # Set properties. Each returns === True on success (i.e. correct resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


    # Watermark
    
    #TODO: Record video
    # save_path = 'saved-media/watermark.mp4'
    # frames_per_seconds = 24
    # config = CFEVideoConf(cap, filepath=save_path, res='720p')
    # out = cv2.VideoWriter(save_path, config.video_type,
    #                       frames_per_seconds, config.dims)
    img_path = 'images/logo/checkin.png'
    logo = cv2.imread(img_path, -1)
    watermark = image_resize(logo, height=80)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
    # end of WM

    img_num=1000


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        #object detect
        resized = cv2.resize(frame, (width, height)) 
        start_time = time.time()
        results = detect_objects(interpreter, resized, args.threshold)
        elapsed_ms = (time.time() - start_time) * 1000
        # end of object detect

        visualize_objects(frame, results, labels)

        # Watermark
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_h, frame_w, frame_c = frame.shape
        # overlay with 4 channels BGR and Alpha
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
        watermark_h, watermark_w, watermark_c = watermark.shape
        # replace overlay pixels with watermark pixel values
        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if watermark[i, j][3] != 0:
                    offset = 10
                    h_offset = frame_h - watermark_h - offset
                    w_offset = frame_w - watermark_w - offset
                    overlay[h_offset + i, w_offset + j] = watermark[i, j]

        cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)
        # End of watermark


        # Display text
        # org
        org = (20, 50)
        # fontScale
        fontScale = 0.7
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Hi! Welcome to student check in system.',
                    org, font, fontScale, color, thickness, cv2.LINE_AA)
        # End of putText()


        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)

            
            if conf >= 30 and conf <= 100:
                # print(5: #id_)
                # print(conf)
                # print(labels[id_])

                if conf >= 80:  
                    print('Student [%s] checked in!' % (name_labels[id_]))
                font = cv2.FONT_HERSHEY_SIMPLEX
                # name = labels[id_]
                namelabel = '%s : %.2f' % (name_labels[id_],conf) + '%'
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, namelabel, (x, y), font, 0.7,
                            color, stroke, cv2.LINE_AA)

            # img_item = "123.png"
            # cv2.imwrite(img_item, roi_color)
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
         
        # Display the resulting frame
        cv2.imshow('Student Check In System', frame)

        #Take snapshot using `s` on the keyboard
        filename = str(img_num) + ".jpg"
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(filename,frame)
            img_num=img_num+1
        #End of taking snap shot

        #Quit the system
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
