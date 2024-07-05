import numpy as np
from sklearn.decomposition import PCA
import bz2
import matplotlib.pyplot as plt
import os
from math import sqrt

######### Reading the files

def read_dm(filename):
    """
    Read word vectors from a word embedding file in a specific format.
    Args:
        filename (str): Path to the word embedding file.
    Returns:
        dict: A dictionary mapping words to their corresponding vectors.
    """
    dm_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            dm_data[word] = vector
    return dm_data

def readDM(dm_file):
    dm_dict = {}
    with open(dm_file) as f:
        dmlines=f.readlines()
    f.close()
    for l in dmlines:
        items=l.rstrip().split()
        row=items[0]
        vec=[float(i) for i in items[1:]]
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def read_bz2(file_path):
    italian_space = {}
    with bz2.open(file_path, 'rt') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            italian_space[word] = vector
    return italian_space

############ Plot figures

def plot_2d_semantic_space(word_vectors, save_path=None, file_name=None):
    """
    Plot a 2D visualization of word vectors using PCA.
    Args:
        word_vectors (dict): A dictionary where keys are words and values are their corresponding vectors.
        save_path (str): Directory path where the plot will be saved. If None, the plot will be displayed interactively.
        file_name (str): Optional file name for saving the plot.
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    words = list(word_vectors.keys())
    vectors = list(word_vectors.values())
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca.fit(vectors)
    reduced_vectors = pca.transform(vectors)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=2, color='blue', alpha=0.5)
    # Annotate each point with its corresponding word
    for i, word in enumerate(words):
        ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=10)
    ax.set_title('2D Semantic Space Visualization')
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    # Save and show the plot based on save_path and file_name
    if save_path:
        if file_name:
            save_path = os.path.join(save_path, file_name)
        plt.savefig(save_path)
        plt.show()
    return fig

def plot_3d_semantic_space(word_vectors, save_path=None, file_name=None):
    """
    Plot a 3D visualization of word vectors using PCA.
    Args:
        word_vectors (dict): A dictionary where keys are words and values are their corresponding vectors.
        save_path (str): Directory path where the plot will be saved. If None, the plot will be displayed interactively.
        file_name (str): Optional file name for saving the plot.
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    words = list(word_vectors.keys())
    vectors = list(word_vectors.values())
    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    pca.fit(vectors)
    reduced_vectors = pca.transform(vectors)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot each word's vector in 3D space
    for word, vector in zip(words, reduced_vectors):
        x, y, z = vector
        ax.scatter(x, y, z, color='b', s=15, edgecolors='k', linewidths=0.5, alpha=0.8)
        ax.text(x, y, z, word, color='black', fontsize=8)
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    # Set title with optional file_name
    if file_name:
        ax.set_title(f'3D Semantic Space Visualization ({file_name})', fontsize=14)
    else:
        ax.set_title('3D Semantic Space Visualization', fontsize=14)
    # Adjust plot limits for better visualization
    max_range = np.array([reduced_vectors[:,0].max()-reduced_vectors[:,0].min(),
                          reduced_vectors[:,1].max()-reduced_vectors[:,1].min(),
                          reduced_vectors[:,2].max()-reduced_vectors[:,2].min()]).max() / 2.0
    mid_x = (reduced_vectors[:,0].max()+reduced_vectors[:,0].min()) * 0.5
    mid_y = (reduced_vectors[:,1].max()+reduced_vectors[:,1].min()) * 0.5
    mid_z = (reduced_vectors[:,2].max()+reduced_vectors[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    # Save or show the plot based on save_path and file_name
    if save_path:
        if file_name:
            save_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"Figure saved in {save_path}")
    return fig

############ PCA, neighbours

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

def run_PCA(dm_dict, words, savefile):
    """
    Run Principal Component Analysis (PCA) on word vectors and plot the 2D representation.
    Args:
        dm_dict (dict): Dictionary where keys are words and values are their corresponding vectors.
        words (list): List of words to include in the PCA.
        savefile (str): File path to save the plot.
    Returns:
        None
    """
    m = []
    labels = []
    for w in words:
        labels.append(w)
        m.append(dm_dict[w])
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca.fit(m)
    m_2d = pca.transform(m)
    # Plot the 2D semantic space and save the plot
    fig = plot_2d_semantic_space(m_2d, labels)
    fig.savefig(savefile)
    plt.close(fig)

def neighbours(dm_dict,vec,n):
    """
   Find the nearest neighbors in dm_dict to a given vector vec based on cosine similarity.
   Args:
       dm_dict (dict): Dictionary where keys are words and values are their corresponding vectors.
       vec (numpy.ndarray): Vector to find neighbors for.
       n (int): Number of neighbors to return.
   Returns:
       list: List of n nearest neighbor words.
   """
    cosines={}
    c=0
    for k,v in dm_dict.items():
        cos = cosine_similarity(vec, v)
        cosines[k]=cos
        c+=1
    c=0
    neighbours = []
    for t in sorted(cosines, key=cosines.get, reverse=True):
        if c<n:
            #print(t,cosines[t])
            neighbours.append(t)
            c+=1
        else:
            break
    return neighbours

def parse_range(range_str):
    """
    Parse a range string and return a range object.
    Args:
        range_str (str): String representing the range in format "start,end".
    Returns:
        range: Range object representing the specified range.
    """
    try:
        n1, n2 = [int(n) for n in range_str.split(',')]
        return range(n1, n2 + 1)  # +1 to include the upper bound
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}")



