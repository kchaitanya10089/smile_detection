# Importing req libs
import cv2
import os

# Defining Paths for dir and haar files
SMILE_CLASSIFIER_FILE_PATH = './smaile_haar_filter.xml'
FACE_CLASSIFIER_FILE_PATH = './face_haar_filter.xml'
INPUT_DIR_PATH = './Input'
OUTPUT_DIR_PATH = './Output'


# Function to load classifiers
def load_classifier():
    try:
        face_classifier = cv2.CascadeClassifier(FACE_CLASSIFIER_FILE_PATH)
        smile_classifier = cv2.CascadeClassifier(SMILE_CLASSIFIER_FILE_PATH)
        if smile_classifier is not None and face_classifier is not None:
            return face_classifier, smile_classifier
        else:
            return None, None
    except Exception as ex:
        print('[ ERROR ]  ', ex)


# Function to process image
def process_image(gray_image, original_image, face_classifier, smile_classifier):
    try:
        # Detecting face from image
        faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)
        for x, y, w, h in faces:
            # Draw face on image
            cv2.rectangle(original_image, (x, y), (w + x, h + y), (255, 0, 255), 2)
            # Cropping face area
            roi_image = gray_image[y:y + h, x:x + w]
            roi_original_image = original_image[y:y + h, x:x + w]
            # Detecting smile from image
            smile = smile_classifier.detectMultiScale(roi_image, 1.7, 8)
            # Drawing smile on original image
            for sx, sy, sw, sh in smile:
                cv2.rectangle(roi_original_image, (sx, sy), (sw + sx, sh + sy), (255, 255, 0), 2)
        print(' [ INFO  ] Processing Completed.!')
        return original_image
    except Exception as ex:
        print('[ ERROR ]  ', ex)


def main():
    try:
        # Loading classifiers into memory
        face_classifier, smile_classifier = load_classifier()
        if smile_classifier is not None and face_classifier is not None:
            # Getting all from from input dir
            for image_name in os.listdir(INPUT_DIR_PATH):
                input_image_path = os.path.join(INPUT_DIR_PATH, image_name)
                output_image_path = os.path.join(OUTPUT_DIR_PATH, image_name)
                original_image = cv2.imread(input_image_path)
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                print('[ INFO ] Processing Image , ', image_name)
                processed_image = process_image(gray_image, original_image, face_classifier, smile_classifier)
                cv2.imwrite(output_image_path, processed_image)
        else:
            print('[ ERROR ] Not able to load classifier.!!')
    except Exception as ex:
        print('[ ERROR ]  ', ex)
