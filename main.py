"""

Usage:
  main.py --nns_range=<n1,n2> --ncomps_range=<n1,n2>
          [--eng_ita | --ita_eng | --eng_cat | --cat_eng | --cat_ita | --ita_cat]
          [-v | --verbose]
          [-s | --save]
          [--Kfold]
  main.py (-h | --help)

Options:
  --ncomps_range=<n1,n2>  Range of principal components to consider (inclusive).
  --nns_range=<n1,n2>     Range of neighbors to evaluate (inclusive).
  --eng_ita               Perform English to Italian translation.
  --ita_eng               Perform Italian to English translation.
  --eng_cat               Perform English to Catalan translation.
  --cat_eng               Perform Catalan to English translation.
  --cat_ita               Perform Catalan to Italian translation.
  --ita_cat               Perform Italian to Catalan translation.
  -v --verbose            Increase output verbosity.
  -s --save               Save the nearest neighbors output in a file.
  -h --help               Show this screen.
  --Kfold                 Use K-fold cross-validation instead of manual train-test selection.

Examples:
  main.py --nns_range=20,40 --ncomps_range=1,6 --eng_ita
  main.py --nns_range=20,35 --ncomps_range=1,5 --ita_cat -s --Kfold
  main.py --nns_range=25,40 --ncomps_range=2,3 --eng_cat -v -s

"""

from docopt import docopt
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import utils
import os
from sklearn.metrics import accuracy_score, f1_score

################

def mk_training_matrices(pairs, dimension_1, dimension_2, space_1, space_2):
    """
    Create training matrices for PLS regression.
    Args:
        pairs (list of str): List of word pairs.
        dimension_1 (int): Dimension of the first space.
        dimension_2 (int): Dimension of the second space.
        space_1 (dict): Semantic space dictionary for the first language.
        space_2 (dict): Semantic space dictionary for the second language.
    Returns:
        tuple: Two numpy arrays, mat_1 and mat_2, representing the training matrices.
    """
    mat_1 = np.zeros((len(pairs), dimension_1))  # Initialize matrix for the first space
    mat_2 = np.zeros((len(pairs), dimension_2))  # Initialize matrix for the second space
    for idx, p in enumerate(pairs):
        word_1, word_2 = p.split()  # Split the word pair into two words
        mat_1[idx] = space_1[word_1]  # Assign the vector of the first word to mat_1
        mat_2[idx] = space_2[word_2]  # Assign the vector of the second word to mat_2
    return mat_1, mat_2

def PLSR(mat_1, mat_2, ncomps):
    """
    Perform Partial Least Squares Regression.
    Args:
        mat_1 (numpy array): Training matrix for the first language.
        mat_2 (numpy array): Training matrix for the second language.
        ncomps (int): Number of components for PLS regression.
    Returns:
        PLSRegression: Trained PLS regression model.
    """
    plsr = PLSRegression(n_components=ncomps, max_iter=4000)  # Initialize PLS regression model
    plsr.fit(mat_1, mat_2)  # Fit the model to the training matrices
    return plsr

def evaluate_plsr(plsr, test_pairs, space_1, space_2, nns, save):
    """
    Function to evaluate the PLS regression model.
    Args:
        plsr (PLSRegression): Trained PLS regression model.
        test_pairs (list of str): List of test word pairs.
        space_1 (dict): Semantic space dictionary for the first language.
        space_2 (dict): Semantic space dictionary for the second language.
        nns (int): Number of nearest neighbors to consider.
        save (bool): Whether to save the results.
    Returns:
        tuple: Adjusted precision, precision, accuracy, F1 score, and results.
    """
    score = 0
    actual_values = []
    predicted_values = []
    results = []
    true_labels = []
    predicted_labels = []
    for p in test_pairs:
        word1, word2 = p.split()
        if word1 not in space_1 or word2 not in space_2:
            continue
        predicted_vector = plsr.predict(space_1[word1].reshape(1, -1))[0]  # Predict the vector for word1
        actual_values.append(space_2[word2])  # Append the actual vector for word2
        predicted_values.append(predicted_vector)  # Append the predicted vector
        nearest_neighbours = utils.neighbours(space_2, predicted_vector, nns)  # Find nearest neighbors in space_2
        if word2 in nearest_neighbours:
            score += 1  # Increment score if the actual word is among the nearest neighbors
            true_labels.append(1)
            predicted_labels.append(1)
            if save:
                results.append([word1, word2, nns] + nearest_neighbours + ["1"])  # Save result with a success indicator
        else:
            true_labels.append(1)
            predicted_labels.append(0)
            if save:
                results.append([word1, word2, nns] + nearest_neighbours + ["0"])  # Save result with a failure indicator
    precision = score / len(test_pairs)  # Calculate precision
    adjusted_precision = precision / nns  # Calculate adjusted precision (depending on nns)
    # Calculate additional metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return adjusted_precision, precision, accuracy, f1, results

def save_results(path, filename, header, results, encoding=None):
    """
    Save results to a file, creating the directory if it doesn't exist.
    This function works just if the user selects the option.
    Args:
        path (str): Directory path where the file will be saved.
        filename (str): Name of the file to save the results.
        header (list of str): List of column names for the header.
        results (list of list): List of results to be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)  # Construct the full file path
    with open(full_path, 'w', encoding=encoding) as f:
        f.write(','.join(header) + '\n')
        for line in results:
            f.write(','.join(map(str, line)) + '\n')

def find_best_parameters(all_results):
    """
    Find the best parameters based on adjusted precision (with nns consideration), precision,
    accuracy, and F1 score.
    Args:
        all_results (list of tuple): List of tuples containing evaluation metrics.
    Returns:
        tuple: Four tuples representing the best parameters by adjusted precision, precision,
               accuracy, and F1 score.
    """
    best_by_adjusted_precision = max(all_results, key=lambda x: x[0])
    best_by_precision = max(all_results, key=lambda x: x[1])
    best_by_accuracy = max(all_results, key=lambda x: x[4])
    best_by_f1 = max(all_results, key=lambda x: x[5])
    return best_by_adjusted_precision, best_by_precision, best_by_accuracy, best_by_f1

def plot_actual_vs_predicted(actual_values, predicted_values, title):
    """
    Plot actual vs. predicted values for two principal components.
    Args:
        actual_values (numpy array): Actual values.
        predicted_values (numpy array): Predicted values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values[:, 0], predicted_values[:, 0], alpha=0.5, label='Component 1')
    plt.scatter(actual_values[:, 1], predicted_values[:, 1], alpha=0.5, label='Component 2', color='red')
    plt.plot([min(actual_values[:, 0]), max(actual_values[:, 0])], [min(actual_values[:, 0]), max(actual_values[:, 0])], color='blue', linestyle='--')  # y=x line for Component 1
    plt.plot([min(actual_values[:, 1]), max(actual_values[:, 1])], [min(actual_values[:, 1]), max(actual_values[:, 1])], color='blue', linestyle='--')  # y=x line for Component 2
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = docopt(__doc__, version='PLSR regression for word translation')
    nns_range = utils.parse_range(args['--nns_range'])
    ncomps_range = utils.parse_range(args['--ncomps_range'])
    save = args['--save']
    verbose = args['--verbose']

    # Setting specific directories for each translation set

    if args['--eng_ita']:
        space_1 = utils.readDM("./spaces/english_space.dm")
        space_2 = utils.read_bz2("./spaces/ita_space.bz2")
        pairs_path = "./pairs/eng_ita_pairs.txt"
        dimension_1 = 400
        dimension_2 = 300

    if args['--ita_eng']:
        space_1 = utils.read_bz2("./spaces/ita_space.bz2")
        space_2 = utils.readDM("./spaces/english_space.dm")
        pairs_path = "./pairs/ita_eng_pairs.txt"
        dimension_1 = 300
        dimension_2 = 400

    if args['--eng_cat']:
        space_1 = utils.readDM("./spaces/english_space.dm")
        space_2 = utils.readDM("./spaces/catalan.subset.dm")
        pairs_path = "./pairs/eng_cat_pairs.txt"
        dimension_1 = 400
        dimension_2 = 300

    if args['--cat_eng']:
        space_1 = utils.readDM("./spaces/catalan.subset.dm")
        space_2 = utils.readDM("./spaces/english_space.dm")
        pairs_path = "./pairs/cat_eng_pairs.txt"
        dimension_1 = 300
        dimension_2 = 400

    if args['--cat_ita']:
        space_1 = utils.readDM("./spaces/catalan.subset.dm")
        space_2 = utils.read_bz2("./spaces/ita_space.bz2")
        pairs_path = "./pairs/cat_ita_pairs.txt"
        dimension_1 = 300
        dimension_2 = 300

    if args['--ita_cat']:
        space_1 = utils.read_bz2("./spaces/ita_space.bz2")
        space_2 = utils.readDM("./spaces/catalan.subset.dm")
        pairs_path = "./pairs/ita_cat_pairs.txt"
        dimension_1 = 300
        dimension_2 = 300

    all_pairs = []
    with open(pairs_path) as f:
        for l in f:
            l = l.rstrip('\n')
            all_pairs.append(l)
    all_results = []

    # The training regime is settled such to be performed with manual selection of training and test sets as well as with 10-fold cross-validation
    if args['--Kfold']:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Initialize KFold with 10 splits
        for nns in nns_range: # Iterate throughout the nns and ncomps combination
            for ncomps in ncomps_range:
                fold_results = []  # List to store results for each fold
                for fold_idx, (train_index, test_index) in enumerate(kf.split(all_pairs), 1):
                    train_pairs = [all_pairs[i] for i in train_index]  # Get training pairs for the current fold
                    test_pairs = [all_pairs[i] for i in test_index]  # Get testing pairs for the current fold
                    mat_1, mat_2 = mk_training_matrices(train_pairs, dimension_1, dimension_2, space_1, space_2)
                    plsr = PLSR(mat_1, mat_2, ncomps)
                    adjusted_precision, precision, accuracy, f1, fold_result = evaluate_plsr(plsr, test_pairs, space_1, space_2, nns, save)
                    fold_results.append((adjusted_precision, precision, nns, ncomps, accuracy, f1, fold_idx))  # Save results for the current fold
                mean_precision = np.mean([r[1] for r in fold_results])  # Calculate mean precision across folds
                adjusted_precision = mean_precision / nns  # Calculate adjusted precision
                all_results.append((adjusted_precision, mean_precision, nns, ncomps, accuracy, f1, fold_idx))  # Save overall results
    else:
        train_pairs = all_pairs[:43] # Manually split the data into training and testing sets
        test_pairs = all_pairs[43:]
        mat_1, mat_2 = mk_training_matrices(train_pairs, dimension_1, dimension_2, space_1, space_2)
        for nns in nns_range:  # Loop over the selected range of nearest neighbors (nns) and components (ncomps)
            for ncomps in ncomps_range:
                plsr = PLSR(mat_1, mat_2, ncomps)
                adjusted_precision, precision, accuracy, f1, results = evaluate_plsr(plsr, test_pairs, space_1, space_2, nns, save)
                all_results.append((adjusted_precision, precision, nns, ncomps, accuracy, f1, results))  # Store the evaluation results for this combination of nns and ncomps
                if verbose:
                    print(f"Precision PLSR (Italian): {precision} with {ncomps} components for {nns} nns (Adjusted Precision: {adjusted_precision})")
    best_by_adjusted_precision, best_by_precision, best_by_accuracy, best_by_f1 = find_best_parameters(all_results)

    print(f"Best Adjusted Precision PLSR: {best_by_adjusted_precision[0]} (Precision: {best_by_adjusted_precision[1]}), accuracy {best_by_adjusted_precision[4]} f1 {best_by_adjusted_precision[5]} with {best_by_adjusted_precision[3]} components for {best_by_adjusted_precision[2]} nns")
    print(f"Best Precision PLSR: {best_by_precision[1]} (Adjusted Precision: {best_by_precision[0]}), accuracy {best_by_precision[4]} f1 {best_by_precision[5]} with {best_by_precision[3]} components for {best_by_precision[2]} nns")
    print("Results and plots will be based on adjusted precision PLSR")

    # Store the best number of components and neighbors based on adjusted precision
    best_ncomps = best_by_adjusted_precision[3]
    best_nns = best_by_adjusted_precision[2]

    if save:
        header = ["Input", "Gold Standard", 'nns_number'] + [f"nns{i}" for i in range(1, best_nns + 1)] + ["Result"]

    # Perform plsr training and testing both with manual selected train-test sets and kfold with the best number of parameters
    if args['--Kfold']:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        actual_values = []
        predicted_values = []
        for train_index, test_index in kf.split(all_pairs):
            train_pairs = [all_pairs[i] for i in train_index]
            test_pairs = [all_pairs[i] for i in test_index]
            mat_1_fold, mat_2_fold = mk_training_matrices(train_pairs, dimension_1, dimension_2, space_1, space_2)
            final_plsr = PLSR(mat_1_fold, mat_2_fold, best_ncomps)
            adjusted_precision, precision, accuracy, f1, results = evaluate_plsr(final_plsr, test_pairs, space_1, space_2, best_nns, save)
            all_results.append((adjusted_precision, precision, best_nns, best_ncomps, accuracy, f1))
            if save:
                output_filename = f'output_nns{nns}_ncomps{ncomps}.csv'
                save_results('./results_bestncomps/manual_selection/', output_filename, header, results, encoding='utf-8')
            actual_values_fold = np.array([space_2[p.split()[1]] for p in test_pairs if p.split()[1] in space_2])
            predicted_values_fold = np.array([final_plsr.predict(space_1[p.split()[0]].reshape(1, -1))[0] for p in test_pairs if p.split()[0] in space_1])
            actual_values.extend(actual_values_fold)
            predicted_values.extend(predicted_values_fold)
        actual_values = np.array(actual_values)
        predicted_values = np.array(predicted_values)
        plot_actual_vs_predicted(actual_values, predicted_values, f'PLSR with {best_ncomps} components for {best_nns} nns (K-fold)'.encode('utf-8'))
    else:
        mat_1, mat_2 = mk_training_matrices(train_pairs, dimension_1, dimension_2, space_1, space_2)
        final_plsr = PLSR(mat_1, mat_2, best_ncomps)
        actual_values = np.array([space_2[p.split()[1]] for p in test_pairs if p.split()[1] in space_2])
        predicted_values = np.array([final_plsr.predict(space_1[p.split()[0]].reshape(1, -1))[0] for p in test_pairs if p.split()[0] in space_1])
        plot_actual_vs_predicted(actual_values, predicted_values, f'PLSR with {best_ncomps} component for {best_nns} nns')
        adjusted_precision, precision, accuracy, f1, results = evaluate_plsr(final_plsr, test_pairs, space_1, space_2, best_nns, save)
        all_results.append((adjusted_precision, precision, best_nns, best_ncomps, accuracy, f1)) # Store the evaluation results for this combination of nns and ncomps
        if save:
            output_filename = f'output_nns{best_nns}_ncomps{best_ncomps}.csv'
            save_results('./results_bestncomps/manual_selection/', output_filename, header, results, encoding='utf-8')


