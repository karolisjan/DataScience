# Detection of faces and facial landmarks

## Requirements

- [dlib 19.4.0](https://pypi.python.org/pypi/dlib) 
- [OpenCV 3](https://pypi.python.org/pypi/opencv-python)
- [scikit-learn](http://scikit-learn.org/stable/)

## Usage

```
usage: detector.py [-h]
                   [--facial_landmarks_predictor_path FACIAL_LANDMARKS_PREDICTOR_PATH]
                   [--gray_scale] [--clahe] [--mark_landmarks]
                   [--mark_landmarks_numbers] [--save] [--resize RESIZE]
                   img_path

positional arguments:
  img_path              path to an image

optional arguments:
  -h, --help            show this help message and exit
  --facial_landmarks_predictor_path FACIAL_LANDMARKS_PREDICTOR_PATH
                        path to shape_predictor_68_face_landmarks.dat
  --gray_scale          converts image into gray_scale
  --clahe               applies Contrast Limited Adaptive Histogram
                        Equalization
  --mark_landmarks      displays the numbers on the landmarks
  --mark_landmarks_numbers
                        displays the numbers on the landmarks
  --save                Saves detected 'faces' from an 'img' in the
                        'img_path'. All detected 'faces' are saved as img_path
                        + '_face_' + str(face_number) + img_extension.
  --resize RESIZE       Resize detected 'faces' before saving them
```

