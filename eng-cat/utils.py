import numpy as np
from sklearn.decomposition import PCA
import bz2
import matplotlib.pyplot as plt

######### Reading the files

def read_dm(filename):
    """
    Read a DM file and create a dictionary of word vectors.
    Args:
    - filename (str): Path to the DM file.
    Returns:
    - dict: Dictionary where keys are words and values are their corresponding vectors.
    """
    dm_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            dm_data[word] = vector
    return dm_data

def read_bz2(file_path):
    """
    Read a bz2 compressed file and create a dictionary of word vectors.
    Args:
    - file_path (str): Path to the bz2 file.
    Returns:
    - dict: Dictionary where keys are words and values are their corresponding vectors.
    """
    italian_space = {}
    with bz2.open(file_path, 'rt') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            italian_space[word] = vector
    return italian_space

def get_dimensions(file_path):
    """
    Get the number of vector dimensions from the first line of a bz2 file.
    Args:
    - file_path (str): Path to the bz2 file.
    Returns:
    - int: Number of vector dimensions.
    """
    with bz2.open(file_path, 'rt') as file:
        first_line = file.readline().strip()
        vector_components = first_line.split()[1:]
        dimensions = len(vector_components)
    return dimensions

def plot_2d_semantic_space(m_2d, labels):
    """
    Create a 2D scatter plot of word vectors.
    Args:
    - m_2d (np.ndarray): 2D array of word vectors transformed by PCA.
    - labels (list): List of word labels corresponding to each vector.
    Returns:
    - matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(m_2d[:, 0], m_2d[:, 1], s=2, color='blue', alpha=0.5)
    for i, label in enumerate(labels):
        ax.annotate(label, (m_2d[i, 0], m_2d[i, 1]), fontsize=10)
    ax.set_title('Semantic Map')
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    return fig

def plot_3d_semantic_space(word_vectors):
    """
    Create a 3D scatter plot of word vectors.
    Args:
    - word_vectors (dict): Dictionary where keys are words and values are their corresponding vectors.
    Returns:
    - None
    """
    words = list(word_vectors.keys())
    vectors = list(word_vectors.values())
    pca = PCA(n_components=3)
    pca.fit(vectors)
    reduced_vectors = pca.transform(vectors)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for word, vector in zip(words, reduced_vectors):
        x, y, z = vector
        ax.scatter(x, y, z, color='b', s=15, edgecolors='k', linewidths=0.5, alpha=0.8)
        ax.text(x, y, z, word, color='black', fontsize=8)
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.set_title('3D Semantic Space Visualization', fontsize=14)

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
    plt.show()

def readDM(dm_file):
    dm_dict = {}
    version = ""
    with open(dm_file) as f:
        dmlines=f.readlines()
    f.close()

    #Make dictionary with key=row, value=vector
    for l in dmlines:
        items=l.rstrip().split()
        row=items[0]
        vec=[float(i) for i in items[1:]]
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict
############ Already existing functions

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (np.sqrt(den_a) * np.sqrt(den_b))

def run_PCA(dm_dict, words, savefile):
    m = []
    labels = []
    for w in words:
        labels.append(w)
        m.append(dm_dict[w])
    pca = PCA(n_components=2)
    pca.fit(m)
    m_2d = pca.transform(m)
    fig = plot_2d_semantic_space(m_2d, labels)
    fig.savefig(savefile)
    plt.close(fig)

def neighbours(dm_dict,vec,n):
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
    try:
        n1, n2 = [int(n) for n in range_str.split(',')]
        return range(n1, n2 + 1)  # +1 to include the upper bound
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}")

