import pandas as pd
import os
import numpy as np
import cv2

def preprocess_image(image_path, width, height):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Image not read properly: {image_path}")
        return None

    image = cv2.resize(image, (width, height))

    normalized_image = image / 255.0

    img_array = np.array(normalized_image).flatten()

    return img_array

def preprocess_folder(folder_path, width, height):

    label = os.path.basename(os.path.normpath(folder_path))
    preprocessed_images = []

    for file in os.listdir(folder_path):

        image_path = os.path.join(folder_path, file)

        preprocessed_image = preprocess_image(image_path, width, height)
        
        if preprocessed_image is not None:

            preprocessed_images.append(preprocessed_image)

    df = pd.DataFrame(preprocessed_images)
    df['label'] = label
    
    return df