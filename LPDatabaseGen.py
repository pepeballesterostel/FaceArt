'''
This script selects images from a given path to a Face database and extracts the Landmark points to generate a database.
The database contains the following information: 'Image_id': unique id, 'Image': file name, 'x_y_coordinates': landmark points.

MediaPipe implementation can be found here: https://google.github.io/mediapipe/getting_started/python.html

Author: Pepe Ballesteros
Last update: 03.05.2022
'''

# Imports
import cv2
import mediapipe as mp
from tqdm import tqdm
import pandas as pd
import utils as ut

# Select a Database and indicate the path

def main(dataset_path):
  # Initialization of models 
  mp_face_mesh = mp.solutions.face_mesh

  # Get images form Database
  image_format = ut.get_format(dataset_path)
  images = ut.get_images(dataset_path, image_format)


  # Extraction of the Landmark points
  with mp_face_mesh.FaceMesh( # with func to handle opening and closing resources of the object face_mesh in the class FaceMesh. 
      static_image_mode=True, # False for video input
      max_num_faces=1,
      refine_landmarks=True, # Apply the attention Mesh model (refine detection of eyes, lips)
      min_detection_confidence=0.4) as face_mesh: # trade-off between acc and latency (Ignored if working on Images)

    landmark_list = []
    failed_images = []
    image_id = []
    print('EXTRACTING LANDMARK POINTS FROM {}'.format(dataset_path))
    for idx, file in tqdm(enumerate(images)):
          image = cv2.imread(file)
          w,h,c = image.shape
          results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          if not results.multi_face_landmarks: 
            #print('image {} failed'.format(file))
            failed_images.append(file)
            continue
          image_id.append(idx)
          coordinates_list= []
          for face_landmarks in results.multi_face_landmarks:  
            for landmark in face_landmarks.landmark:
              # append image, coordenates, and path
              coordinates_list.append(landmark.x)
              coordinates_list.append(landmark.y)
            #print('finished calculating all landmarks')
          landmark_list.append(coordinates_list)
          #print('---------------------------------------coordinates for image {} added to the list'.format(file))
    
  target_files = [file for file in images if file not in failed_images]

  # Save the results in a CSV file
  dict = {'Image_id': image_id, 'Image': target_files, 'x_y_coordinates': landmark_list}
  df = pd.DataFrame(dict)
  df.to_csv('outputs/' + dataset_path + '.csv', index = False)
  return df
