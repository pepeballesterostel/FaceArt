'''

Author: Pepe Ballesteros
Last update: 03.05.2022
'''
import os 
import numpy as np
import glob
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import plotly.express as px
from kneed import KneeLocator
import random


# Get the images from the Database
def get_format(dataset_path):
    '''
    Function that randomly selects and image from the database and gets its format
    '''
    images = os.listdir(dataset_path)
    return random.choice(images).split('.')[-1]

def get_images(path, image_format):
    '''
    Function that takes a path and a format and returns a list of images
    '''
    images = []
    for file in glob.glob(os.path.join(path, '*.'+image_format)):
        images.append(file)
    return images

def get_LDP(row):
    '''
    Gets the input df row and converts it: from a str list to an array of floats
    '''
    row = [float(x.strip(' []')) for x in row.split(',')]
    return np.array(row, dtype=np.float32)

def get_matrix(df):
    '''
    Convert the list of floats to a matrix
    '''
    return np.array(df.tolist())

def tsne_plot_3d(title, label, embeddings, df, feature, a=1):
    sns.palplot(sns.color_palette("Set2", 16))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    color_labels = df[feature].unique()
    rgb_values = sns.color_palette("Set2", 16)
    color_map = dict(zip(color_labels, rgb_values))
    #colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c = df[feature].map(color_map), alpha=a, label=label)
    plt.grid(False)
    plt.axis('off')
    plt.title(title)
    plt.show()

def tsne_plot_2d(title, df, hue, alpha=1):
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data = df, x='x_coord', y='y_coord', hue=hue, palette="Set2", alpha=alpha, legend = 'brief')
    plt.grid(False)
    plt.axis('off')
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', borderaxespad=0, title=hue)
    plt.show()

def remove_extension(s):
    '''
    A function that removes the extension from a file name
    '''
    return int('.'.join(s.split('.')[:-1]))

def remove_imagefiles():
    '''
    A function that removes image files from folder
    '''
    for file in glob.glob('*.png'): 
        os.remove(file)

def get_coordinates_columns(df,embeddings):
    '''
    Gets the embeddings and returns a dataframe with the coordinates
    '''
    n, m = embeddings.shape
    df['x_coord'] = [embedding[0] for embedding in embeddings]
    df['y_coord'] = [embedding[1] for embedding in embeddings]
    if m == 3:
        df['z_coord'] = [embedding[2] for embedding in embeddings]
    return df 

def plot_interactive_2d(df):
    fig = px.scatter(df, x='x_coord', y='y_coord', color=df['Style'], log_x=True,
                 hover_name="image_id")
                  #,hover_data=["Artist", "Title"])
    fig.show()

def plot_interactive_2d_kmeans(df):
    fig = px.strip(df, x='x_coord', y='y_coord', color=df['cluster'], log_x=True,
                 hover_name="Image_id", hover_data="Image")
    fig.show()

def plot_interactive_3d(df):
    fig = px.scatter_3d(df, x='x_coord', y='y_coord' ,z='z_coord', color=df['Style'], log_x=True,
                 hover_name="image_id", hover_data=["Artist", "Title"])
    fig.show()

def plot_interactive_3d_kmeans(df):
    fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord', color=df['cluster'], log_x=True,
                 hover_name="Image_id", hover_data='Image')
    fig.show()

def elbow_kmeans(X, n_clusters, n_init=10, init='random', algorithm = 'full', random_state=12):
    '''
    A function that applies the elbow method and returns the optimal number of clusters
    '''
    distortions = []
    for i in range(1, n_clusters):
        kmeans = KMeans(n_clusters=i, init=init, n_init=n_init, algorithm=algorithm, random_state=random_state)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    kl = KneeLocator(range(1, n_clusters), distortions, curve="convex", direction="decreasing")
    return kl.elbow

def k_means(X, df, max_clusters, n_init=10, init='random', algorithm = 'full', random_state=12):
    '''
    A function that applies K-means clustering to columns of a dataframe and returns a dataframe with the cluster labels
    '''
    n_clusters = elbow_kmeans(X,max_clusters)
    print('Optimal number of clusters: {}'.format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, algorithm=algorithm, random_state=random_state)
    kmeans.fit(X)
    labels = kmeans.labels_
    df['cluster'] = labels