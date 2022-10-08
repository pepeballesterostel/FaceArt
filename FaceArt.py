'''
This script launches a web application that captures faces from a webcam or an image and returns 
the most similar face expressions from an art database.

Author: Pepe Ballesteros
Last update: 03.05.2022
'''
import argparse
import sys
import gradio as gr
import mediapipe as mp
import cv2
import numpy as np
from paramiko import Channel
import recommender_system as rs
import LPDatabaseGen

parser = argparse.ArgumentParser(prog = 'FaceArt', description='FaceArt: A face expression recommender system', epilog = 'Enjoy!')
parser.add_argument('-d', '--dataset', type = str, help='Indicate the name of the folder where the images are located (Required)', required=True)
parser.add_argument('-s', '--save', help='Compute the landmark point database of the images and save it in the output folder. Required every time a new image dataset is used', action='store_true') 
parser.add_argument('-i', '--input', type = str, help='Indicate webcam or image. Whether to use the webcam or another image as an input (default is webcam)') 
args = parser.parse_args()

def extract_face(image):
    '''
    Function that takes an input image snapshot form Gradio (numpy array) and returns the cropped image of the face
    '''
    mp_face_detection = mp.solutions.face_detection
    height, width, channel = image.shape
    with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.4) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            annotated_image = image.copy()
            for detection in results.detections:
                bBox = detection.location_data.relative_bounding_box
                x,y,w,h = int(bBox.xmin * width), int(bBox.ymin * height), int(bBox.width * width), int(bBox.height * height)
                aug_factor_x = int(0.2 * w)
                aug_factor_y = int(0.2 * h)
                cropped_image = annotated_image[y-aug_factor_y:y+h+aug_factor_y,x-aug_factor_x:x+w+aug_factor_x]
    return cropped_image

def extract_mesh(image):
    '''
    Function that takes an input image snapshot form Gradio (numpy array) and returns Landmark Points as a np array
    and the mesh of the cropped face with white background
    '''
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    coordinates_list= []
    with mp_face_mesh.FaceMesh( static_image_mode=True, max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.4) as face_mesh:
        w,h,c = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        white_background = np.zeros([w,h,c],dtype=np.uint8)
        white_background.fill(255)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    coordinates_list.append(landmark.x)
                    coordinates_list.append(landmark.y)
                # Drawing
                mp_drawing.draw_landmarks(
                    image= white_background,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
    return np.array(coordinates_list, dtype=np.float32), white_background

def get_recommended_images(files):
    imgs = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

def webapp(image):
    # Locate the face in the input image
    cropped = extract_face(image)
    # Calculate LP
    coordinates, white_background = extract_mesh(cropped)
    # Get recommendations
    recomended = rs.get_recommendations(coordinates, df, n=3)
    files = rs.get_image_names(df, recomended)
    recomendation_images = get_recommended_images(files)
    meshs = rs.get_mesh(files)
    meshs = rs.scale_images(meshs)
    recomendation_images = rs.scale_images(recomendation_images)
    im_vh = rs.concat_vh([[recomendation_images[0], recomendation_images[1], recomendation_images[2]],
                    [meshs[0], meshs[1], meshs[2]]])
    return white_background, im_vh

# Setup variables
SAVE_OUTPUT_CSV = True if args.save else False
dataset_path = args.dataset
user_input = args.input if args.input else 'webcam'
description = """
    Pepe Ballesteros Zapata - Digital Visual Studies
    """

if SAVE_OUTPUT_CSV:
    df = LPDatabaseGen.main(dataset_path)
else:
    df = rs.load_dataset('outputs/' + dataset_path + '.csv')

# Web app Launching
def main():
    if user_input == 'webcam':
        input = gr.inputs.Image(shape=(640,480), source="webcam", type="numpy")
    else:
        input = gr.inputs.Image(type="numpy")

    output1 = gr.outputs.Image(type="numpy", label='Detected Mesh')
    output2 = gr.outputs.Image(type="numpy", label='Top 3 Recommendations')
    app = gr.Interface(fn = webapp, inputs = input , outputs = [output1,output2], title='Face Art', theme = 'dark-peach',
                        description=description, layout='aligned')
    try:
        app.launch(share=True)
    except KeyboardInterrupt:
        app.close()
        sys.exit()

if __name__ == "__main__":
    main()