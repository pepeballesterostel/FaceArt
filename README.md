# FaceArt
<p align="center">
  <img src="https://github.com/pepeballesterostel/FaceArt/blob/main/FaceArt.png" width="200" />
</p>

## Description

This is an implementation of a website that allows you to retrieve similar artworks from a database based on face expression similarity. The system detects the Face Mesh of an input image, and retrieves the top 3 paintings based on landmark points similarity score. The user selects wheter to use the webcam feature to capture the input image, or select an already existing one. The project uses Google's Face detection and Face Mesh models [MediaPipe](https://google.github.io/mediapipe/). The implementation of the Webapp uses [Gradio](https://gradio.app/) Interface.


## Installation

First, create a conda environment to locate all dependencies. Then, you can install the required packages using the requirements.txt file.
```
$ conda env create FaceArt
$ conda activate FaceArt
$ pip install -r requirements.txt
```

## Usage 
To launch the application, run the FaceArt.py file indicating the required arguments. First, type python FaceArt.py -h in the command line to check the arguments:
```
$ python FaceArt.py -h 
usage: FaceArt [-h] -d DATASET [-s] [-i INPUT]

FaceArt: A face expression recommender system

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Indicate the name of the folder where the images are located (Required)
  -s, --save            Compute the landmark point database of the images and save it in the output folder. Required every time a new image dataset is used
  -i INPUT, --input INPUT
                        Indicate webcam or image. Whether to use the webcam or another image as an input (default is webcam)

Enjoy!
```
- **Dataset**: name of the folder where the paintings are located. This repo shares a small sample of images from the WikiArt dataset located in the /examples directory. You can try to run the app with your own images simply by creating a new folder with the target images and giving the name as the -d argument.
- **Save**: This argument indicates the program whether is neccesary to compute the landmark points for the database. This repo already shares the landmark points for the /example database. Use this argument (simply include --save in the comand line) the first time you use a new image dataset. Once computed, a csv file with the landmark information of your database will be saved in the directory /outputs. 
- **Input**: indicate the program if you want to capture similarity live from your webcam, or upload another image. 

## Example

After the installation step, run the application:
```
$ python FaceArt.py --dataset examples --input webcam
```
The following options will pop up: *Running on local URL* and *Running on public URL*. Click one of the two options, and the web application will be launched in the browser.
- **Important**: To end the application, go back to your terminal and press 'Ctrl + C'
