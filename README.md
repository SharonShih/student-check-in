# CMPE 181 Final Group Project
### OpenCV & Python & TensorFlow

## :star:  [See Our Video Demo](https://www.youtube.com/watch?v=Wg96SrO8SrA&feature=youtu.be)


## Steps to run the project

### 1. Requirement/Installation
- Make sure your laptop has webcam
- OpenCV Installation:
  - MacOS: https://www.codingforentrepreneurs.com/blog/install-opencv-3-for-python-on-mac/
  - Windows: https://www.codingforentrepreneurs.com/blog/install-opencv-3-for-python-on-windows
  - Make sure to install `pillow` and `numpy` for import
- Python 3.7 or above
- TensorFlow: https://www.tensorflow.org/lite/guide/python

### 2. Run `faces-train.py`
- In order to start the training, go to `/student-check-in/src/`, and run command:
``` command
python faces.train.py
```
- If you want to add new faces to train, create a new folder in `/student-check-in/src/images`
- And make sure the folder name is the person's name and add images.
- Then rerun `python faces.train.py` whenever you add new images

### 3. Ready to run the system
- Run `python faces.py` to open the web cam and run the project

## Functions you should try
- Able to recognize faces based on trained dataset/model
- Check-in students when the similarity is larger than 80%.
- Able to detect the student ID and recognize the faces on the ID photo.
- Able to save screenshots when the user click the ‘S’ button on the keyboard.
- Able to quit system when the user click the ‘Q’ button.
- Able to detect any mobile devices and show Alert message.


## References:
- Face recognition:
  - https://www.youtube.com/watch?v=PmZ29Vta7Vc&t=602s 
  - https://github.com/codingforentrepreneurs/OpenCV-Python-Series
- Object detection: https://github.com/lkk688/GoogleCloudIoT/tree/master/iotpython


## Team Members
- [En-Ping Shih](https://github.com/SharonShih)
- [Yang Li](https://github.com/liamLacuna)
