import numpy as np
import utils
from sklearn.metrics.pairwise import cosine_similarity
import os

def compute_cosine_similarities(data, space, dataset_name):
    """
    Compute cosine similarities between gold standard translation vector and
    nns vectors for each translated word in the test set using the semantic space.
    Args:
        data (list of lists): Input data containing [Input, Gold Standard, nns_number, nns1, nns2, nns3, Result].
        space (dict): Semantic space where keys are words and values are their vector representations.
        dataset_name (str): Name of the dataset (for printing average average cosine similarity).
    Returns:
        tuple: Processed data with added cosine similarities and average of average cosine similarities across rows.
    """
    processed_data = []
    for row in data:
        word1 = row[1]  # Gold Standard Translation
        nns_words = row[3:6]  # Outputted Nearest neighbors
        if word1 in space:
            vector1 = space[word1]
            vector1 = np.array(vector1)
            vector1 = vector1.reshape(1, -1)
            nns_cosine_sims = []
            for nn_word in nns_words:
                if nn_word in space:
                    nn_vector = space[nn_word]
                    nn_vector = np.array(nn_vector)
                    nn_vector = nn_vector.reshape(1, -1)
                    nn_cosine_sim = cosine_similarity(vector1, nn_vector)
                    nns_cosine_sims.append(nn_cosine_sim[0][0])
                    nn_cosine_sim = nn_cosine_sim[0][0]
                    row.append(nn_cosine_sim)
                else:
                    row.append(0.0)  # Default to 0 if nearest neighbor word not in space
            # Compute average cosine similarity of nns1, nns2, nns3 (since best performances are with nns = 3)
            if len(nns_cosine_sims) >= 3:
                avg_nns_cosine_sim = np.mean(nns_cosine_sims[:3])
            else:
                avg_nns_cosine_sim = 0.0
            row.append(avg_nns_cosine_sim)
            processed_data.append(row)
        else:
            processed_data.append(row + [0.0] * 4)  # Append zeros if word vectors not found
    # Compute average of Avg_Cosine_Sim_nns1_nns2_nns3 for each row (word translation)
    avg_avg_nns_cosine_sim = np.mean([row[-1] for row in processed_data])
    print(dataset_name, avg_avg_nns_cosine_sim)  # Print average average cosine similarity
    return processed_data, avg_avg_nns_cosine_sim

def save_processed_data(data, path, filename):
    """
    Save processed data to a file.
    Args:
        data (list of lists): Processed data containing cosine similarities and averages.
        path (str): Directory path where the file will be saved.
        filename (str): Name of the file to save the results.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    header = ['Input', 'Gold Standard', 'nns_number', 'nns1', 'nns2', 'nns3', 'Result', 'Cosine_Sim_Word1_Word2', 'Cosine_Sim_nns1', 'Cosine_Sim_nns2', 'Cosine_Sim_nns3']
    with open(full_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in data:
            f.write(','.join(map(str, row)) + '\n')

# Semantic Spaces Directories
space_ita = utils.read_bz2('./spaces/ita_space.bz2')
space_eng = utils.readDM('./spaces/english_space.dm')
space_cat = utils.readDM('./spaces/catalan.subset.dm')

if __name__ == '__main__':

    # nns outputs for best performance with nns=3 for each languages pair

    data_set1 = [
        ['black', 'nero', 3, 'rosso', 'bianco', 'nero', 1],
        ['short', 'breve', 3, 'questo', 'lungo', 'piccolo', 0],
        ['day', 'giorno', 3, 'notte', 'giorno', 'anno', 1],
        ['five', 'cinque', 3, 'tre', 'quattro', 'due', 0],
        ['tongue', 'lingua', 3, 'bambino', 'testa', 'mano', 0],
        ['few', 'pochi', 3, 'due', 'tre', 'quattro', 0],
        ['big', 'grande', 3, 'piccolo', 'vicino', 'vecchio', 0],
        ['earth', 'terra', 3, 'mare', 'fuoco', 'terra', 1],
        ['lake', 'lago', 3, 'fiume', 'mare', 'lago', 1],
        ['when', 'quando', 3, 'che', 'questo', 'quando', 1],
        ['other', 'altro', 3, 'due', 'alcuni', 'molti', 0],
        ['sun', 'sole', 3, 'stella', 'sole', 'notte', 1],
        ['who', 'chi', 3, 'lui', 'che', 'lei', 0],
        ['he', 'lui', 3, 'lei', 'lui', 'marito', 1],
    ]

    eng_ita = compute_cosine_similarities(data_set1, space_ita, 'eng_ita')
    save_processed_data(eng_ita, './results_bestncomps/cosine_similarity/', 'eng_ita.csv')

    data_set2 = [
        ['nero', 'black', 3, 'white', 'red', 'black', 1],
        ['breve', 'short', 3, 'long', 'short', 'small', 1],
        ['giorno', 'day', 3, 'night', 'year', 'day', 1],
        ['cinque', 'five', 3, 'three', 'four', 'two', 0],
        ['lingua', 'tongue', 3, 'mother', 'river', 'many', 0],
        ['pochi', 'few', 3, 'many', 'some', 'few', 1],
        ['grande', 'big', 3, 'small', 'how', 'big', 1],
        ['terra', 'earth', 3, 'sea', 'star', 'they', 0],
        ['lago', 'lake', 3, 'river', 'sea', 'lake', 1],
        ['quando', 'when', 3, 'where', 'when', 'that', 1],
        ['altro', 'other', 3, 'this', 'that', 'not', 0],
        ['sole', 'sun', 3, 'star', 'night', 'straight', 0],
        ['chi', 'who', 3, 'not', 'that', 'person', 0],
        ['lui', 'he', 3, 'father', 'mother', 'he', 1],
    ]

    ita_eng = compute_cosine_similarities(data_set2, space_eng, 'ita_eng')
    save_processed_data(ita_eng, './results_bestncomps/cosine_similarity/', 'ita_eng.csv')

    data_set3 = [
        ['negre', 'black', 3, 'red', 'white', 'black', 1],
        ['curt', 'short', 3, 'thick', 'long', 'short', 1],
        ['dia', 'day', 3, 'night', 'year', 'day', 1],
        ['cinc', 'five', 3, 'four', 'three', 'two', 0],
        ['llengua', 'tongue', 3, 'person', 'woman', 'mother', 0],
        ['pocs', 'few', 3, 'many', 'two', 'few', 1],
        ['gran', 'big', 3, 'small', 'thick', 'thin', 0],
        ['terra', 'earth', 3, 'salt', 'sea', 'water', 0],
        ['llac', 'lake', 3, 'river', 'sea', 'lake', 1],
        ['quan', 'when', 3, 'what', 'where', 'when', 1],
        ['altre', 'other', 3, 'small', 'this', 'new', 0],
        ['sol', 'sun', 3, 'fire', 'straight', 'thick', 0],
        ['qui', 'who', 3, 'father', 'mother', 'what', 0],
        ['ell', 'he', 3, 'they', 'he', 'father', 1],
    ]

    cat_eng = compute_cosine_similarities(data_set3, space_eng, 'cat_eng')
    save_processed_data(cat_eng, './results_bestncomps/cosine_similarity/', 'cat_eng.csv')

    data_set4 = [
        ['black', 'negre', 3, 'blanc', 'vermell', 'negre', 1],
        ['short', 'curt', 3, 'llarg', 'curt', 'peu', 1],
        ['day', 'dia', 3, 'any', 'nit', 'dia', 1],
        ['five', 'cinc', 3, 'tres', 'quatre', 'cinc', 1],
        ['tongue', 'llengua', 3, 'mare', 'no', 'mà', 0],
        ['few', 'pocs', 3, 'dos', 'tres', 'molts', 0],
        ['big', 'gran', 3, 'petit', 'estret', 'gruixut', 0],
        ['earth', 'terra', 3, 'mar', 'tots', 'un', 0],
        ['lake', 'llac', 3, 'riu', 'mar', 'llac', 1],
        ['when', 'quan', 3, 'on', 'què', 'quan', 1],
        ['other', 'altre', 3, 'molts', 'dos', 'tots', 0],
        ['sun', 'sol', 3, 'estrella', 'dona', 'nit', 0],
        ['who', 'qui', 3, 'qui', 'ell', 'ells', 1],
        ['he', 'ell', 3, 'qui', 'ell', 'pare', 1],
    ]

    eng_cat = compute_cosine_similarities(data_set4, space_cat, 'eng_cat')
    save_processed_data(eng_cat, './results_bestncomps/cosine_similarity/', 'eng_cat.csv')

    data_set5 = [
        ['negre', 'nero', 3, 'bianco', 'rosso', 'nero', 1],
        ['curt', 'breve', 3, 'spesso', 'lungo', 'questo', 0],
        ['dia', 'giorno', 3, 'notte', 'giorno', 'anno', 1],
        ['cinc', 'cinque', 3, 'tre', 'quattro', 'due', 0],
        ['llengua', 'lingua', 3, 'persona', 'spesso', 'cosa', 0],
        ['pocs', 'pochi', 3, 'molti', 'alcuni', 'pochi', 1],
        ['gran', 'grande', 3, 'piccolo', 'un', 'grande', 1],
        ['terra', 'terra', 3, 'mare', 'sale', 'notte', 0],
        ['llac', 'lago', 3, 'fiume', 'lago', 'mare', 1],
        ['quan', 'quando', 3, 'che', 'lei', 'quando', 1],
        ['altre', 'altro', 3, 'alcuni', 'nuovo', 'questo', 0],
        ['sol', 'sole', 3, 'spesso', 'fuoco', 'essi', 0],
        ['qui', 'chi', 3, 'lei', 'lui', 'madre', 0],
        ['ell', 'lui', 3, 'lui', 'lei', 'donna', 1],
    ]

    cat_ita = compute_cosine_similarities(data_set5, space_ita, 'cat_ita')
    save_processed_data(cat_ita, './results_bestncomps/cosine_similarity/', 'cat_ita.csv')

    data_set6 = [
        ['nero', 'negre', 3, 'blanc', 'vermell', 'negre', 1],
        ['breve', 'curt', 3, 'llarg', 'curt', 'petit', 1],
        ['giorno', 'dia', 3, 'any', 'nit', 'dia', 1],
        ['cinque', 'cinc', 3, 'tres', 'quatre', 'cinc', 1],
        ['lingua', 'llengua', 3, 'molts', 'persona', 'pocs', 0],
        ['pochi', 'pocs', 3, 'algun', 'molts', 'pocs', 1],
        ['grande', 'gran', 3, 'petit', 'vell', 'gran', 1],
        ['terra', 'terra', 3, 'mar', 'terra', 'molts', 1],
        ['lago', 'llac', 3, 'riu', 'mar', 'llac', 1],
        ['quando', 'quan', 3, 'on', 'quan', 'què', 1],
        ['altro', 'altre', 3, 'aquest', 'altre', 'aquell', 1],
        ['sole', 'sol', 3, 'estrella', 'nit', 'cel', 0],
        ['chi', 'qui', 3, 'què', 'qui', 'no', 1],
        ['lui', 'ell', 3, 'qui', 'ell', 'què', 1],
    ]

    ita_cat = compute_cosine_similarities(data_set6, space_cat, 'ita_cat')
    save_processed_data(ita_cat, './results_bestncomps/cosine_similarity/', 'ita_cat.csv')