"""
Usage:
  data_handling.py [--words_list [--eng | --ita | --cat]]
                   [--plot [--ita | --eng | --cat]]
  data_handling.py (-h | --help)

Options:
  --words_list       Output a file with words from the specified semantic space
    --eng            English
    --ita            Italian
    --cat            Catalan
  --plot             Output 2D and 3D visualizations of semantic space subsets for translations
    --ita_cat        Italian to/from Catalan
    --eng_cat        English to/from Catalan
    --eng_ita        English to/from Italian
  -h, --help         Show this help message

Examples:
  data_handling.py --words_list --eng
  data_handling.py --plot --ita_cat
  data_handling.py --words_list --eng --plot --ita_cat

"""

import re
import bz2
from docopt import docopt
import numpy as np
import utils

########### Functions

########### Words lists

def extract_words_from_dm(filename):
    dm_data = utils.read_dm(filename)
    words = list(dm_data.keys())
    return words

def extract_words_from_bz2(filename):
    word_pattern = re.compile(r'\b[A-Za-z]+\b')  # Regular expression to match words
    with bz2.open(filename, 'rt') as file:
        text = file.read()
    words = word_pattern.findall(text)
    return words

def write_words_to_file(words, output_file):
    with open(output_file, 'w') as file:
        for word in words:
            file.write(word + '\n')

########### Plots

def extract_first_words(input_file):
    """
    Extract the first words from each line in the input file.
    Args:
        input_file (str): Path to the input file.
    Returns:
        list: List of first words extracted from each line.
    """
    first_words = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        words = line.split()
        if len(words) >= 1:
            first_words.append(words[0])
        else:
            print(f"Warning: Line '{line.strip()}' does not contain any words.")
    return first_words

def extract_word_vectors(semantic_space, words_of_interest):
    """
    Extract word vectors from a semantic space dictionary for specified words of interest.
    Args:
        semantic_space (dict): Dictionary where keys are words and values are their corresponding vectors.
        words_of_interest (list): List of words for which vectors should be extracted.
    Returns:
        dict: Dictionary where keys are words of interest and values are their corresponding vectors.
    """
    word_vectors = {}
    for word in words_of_interest:
        if word in semantic_space:
            word_vectors[word] = np.array(semantic_space[word])
        else:
            print(f"Warning: '{word}' not found in the semantic space.")
    return word_vectors

############# Main

if __name__ == '__main__':
    args = docopt(__doc__, version='PLSR regression for word translation')

    if args['--words_list']:
        # English space
        if args['--eng']:
            filename = './spaces/english_space.dm'
            output_filename = './pairs/sets/eng_words.txt'
            words_1 = extract_words_from_dm(filename)

        # Italian space
        if args['--ita']:
            filename = './spaces/ita_space.bz2'
            output_filename = './pairs/sets/ita_words.txt'
            words_1 = extract_words_from_bz2(filename)

        # Catalan space
        if args['--cat']:
            filename = './spaces/catalan.subset.dm'
            output_filename = './pairs/sets/cat_words.txt'
            words_1 = extract_words_from_dm(filename)
        write_words_to_file(words_1, output_filename)

    # Plot 2D and 3D images of the specified subset semantic spaces based on PCA
    if args['--plot']:
        if args['--ita']:
            first_words = extract_first_words('./pairs/ita_cat_pairs.txt')
            ita_space_subset = extract_word_vectors(utils.read_bz2('./spaces/ita_space.bz2'), first_words)
            utils.plot_2d_semantic_space(ita_space_subset, save_path='./spaces/figures/', file_name = 'ita_space_subset_2D')
            utils.plot_3d_semantic_space(ita_space_subset, save_path='./spaces/figures/', file_name = 'ita_space_subset_3D')

        if args['--eng']:
            first_words = extract_first_words('./pairs/eng_cat_pairs.txt')
            eng_space_subset = extract_word_vectors(utils.read_dm('./spaces/english_space.dm'), first_words)
            utils.plot_2d_semantic_space(eng_space_subset, save_path='./spaces/figures/', file_name = 'eng_space_subset_2D')
            utils.plot_3d_semantic_space(eng_space_subset, save_path='./spaces/figures/', file_name = 'eng_space_subset_3D')

        if args['--cat']:
            first_words = extract_first_words('./pairs/cat_ita_pairs.txt')
            cat_space_subset = extract_word_vectors(utils.read_dm('./spaces/catalan.subset.dm'), first_words)
            utils.plot_2d_semantic_space(cat_space_subset, save_path='./spaces/figures/', file_name = 'cat_space_subset_2D')
            utils.plot_3d_semantic_space(cat_space_subset, save_path='./spaces/figures/', file_name = 'cat_space_subset_3D')
