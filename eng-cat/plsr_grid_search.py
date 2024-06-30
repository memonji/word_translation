"""Word translation

Usage:
  plsr_grid.py --nns=<n> [-v | --verbose]
  plsr_grid.py (-h | --help)
  plsr_grid.py --version

Options:
  --nns=<n>      Number of nearest neighbours for the evaluation
  -h --help      Show this screen.
  --version      Show version.
  -v --verbose   Show verbose output.

"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import utils
from docopt import docopt

def mk_training_matrices(pairs, en_dimension, cat_dimension, english_space, catalan_space):
    en_mat = np.zeros((len(pairs), en_dimension))
    cat_mat = np.zeros((len(pairs), cat_dimension))
    for c, p in enumerate(pairs):
        en_word, cat_word = p.split()
        en_mat[c] = np.array(english_space[en_word])  # Convert to NumPy array
        cat_mat[c] = np.array(catalan_space[cat_word])  # Convert to NumPy array
    return en_mat, cat_mat

def PLSR(mat_english, mat_catalan, ncomps):
    plsr = PLSRegression(n_components=ncomps)
    plsr.fit(mat_english, mat_catalan)
    return plsr

if __name__ == '__main__':
    args = docopt(__doc__, version='PLSR regression for word translation 1.1')
    nns = int(args["--nns"])
    verbose = args["--verbose"]

    # Read semantic spaces
    english_space = utils.read_dm("data/english.subset.dm")
    catalan_space = utils.read_dm("data/catalan.subset.dm")

    # Read all word pairs
    with open("data/pairs.txt") as f:
        all_pairs = [line.rstrip('\n') for line in f]

    # Make training/test split
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)

    # Make training/test matrices
    en_mat, cat_mat = mk_training_matrices(train_pairs, 400, 300, english_space, catalan_space)

    # Define initial parameter grid
    param_grid = {'n_components': list(range(1, 31))}  # Initial broad search range

    # Perform grid search
    plsr = PLSRegression()
    grid_search = GridSearchCV(plsr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(en_mat, cat_mat)

    best_ncomps = grid_search.best_params_['n_components']
    print(f"Best number of components: {best_ncomps}")

    # Train the final model with the refined best parameter
    plsr = PLSR(en_mat, cat_mat, best_ncomps)

    # Predict with PLSR
    score = 0
    results = []
    for p in test_pairs:
        en, cat = p.split()
        en_vector = np.array(english_space[en])  # Convert to NumPy array
        predicted_vector = plsr.predict(en_vector.reshape(1, -1))[0]
        nearest_neighbours = utils.neighbours(catalan_space, predicted_vector, nns)
        if cat in nearest_neighbours:
            score += 1
            if verbose:
                print(en, cat, nearest_neighbours, "1")
        else:
            if verbose:
                print(en, cat, nearest_neighbours, "0")
        precision = score / len(test_pairs)
        results.append((precision, best_ncomps))
    print("Precision PLSR:", score / len(test_pairs))

    highest_precision, best_ncomps = max(results, key=lambda x: x[0])
    print(f"Highest Precision PLSR: {highest_precision} with {best_ncomps} components for {nns} nns")

