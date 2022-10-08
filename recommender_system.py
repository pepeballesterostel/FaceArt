'''

Author: Pepe Ballesteros
Last update: 03.05.2022
'''
import pandas as pd
import utils
import cv2
import numpy as np
import mediapipe as mp
import utils

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df['x_y_coordinates'] = df['x_y_coordinates'].apply(utils.get_LDP)
    return df
    
def cosine_similarity(v1,v2):
    '''
    A function that computes the cosine similarity between 2 input vectors
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_recommendations(input, df, n=5):
    '''
    A function that takes a df and a vector, calculates the cosine similarity on the column 'df.x_y_coordinates' of 
    the df and returns the top 5 similar embeddings based on 'Image_id'
    '''
    cosine_similarities = []
    for i in range(len(df)):
        # Compute cosine similarity
        cosine_similarities.append(cosine_similarity(input, df['x_y_coordinates'].iloc[i]))
    cosine_similarities = pd.DataFrame(cosine_similarities)
    cosine_similarities.columns = ['cosine_similarity']
    cosine_similarities['Image_id'] = df['Image_id']
    cosine_similarities = cosine_similarities.sort_values(by='cosine_similarity', ascending=False)
    return cosine_similarities.head(n)

def get_image_names(df,recomended):
    '''
    Gets the image names from the recomended df
    '''
    files = []
    for i in range(len(recomended)):
        file = df.loc[df['Image_id'] == recomended.Image_id.iloc[i]]
        files.append(file.Image.astype(str).iloc[0])
    return files


def scale_images(imgs):
    '''
    This function scales the images to same size (max height and max width) for concatenation
    '''
    max_height = 0
    max_width = 0
    for i in range(len(imgs)):
        if imgs[i].shape[0] > max_height:
            max_height = imgs[i].shape[0]
        if imgs[i].shape[1] > max_width:
            max_width = imgs[i].shape[1]
    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (max_width, max_height))
    return imgs

def get_mesh(files, WHITE_BACKGROUND = True):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    meshs = []
    for file in files:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,refine_landmarks=True, 
            min_detection_confidence=0.4) as face_mesh: 
            image = cv2.imread(file)
            w,h,c = image.shape
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                pass
            annotated_image = image.copy()
            white_background = np.zeros([w,h,c],dtype=np.uint8)
            white_background.fill(255) 
            for face_landmarks in results.multi_face_landmarks: 
                background = white_background if WHITE_BACKGROUND else annotated_image         
                # Drawing
                mp_drawing.draw_landmarks(
                    image= background,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
            meshs.append(background)
    return meshs

def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) 
                        for list_h in list_2d])











