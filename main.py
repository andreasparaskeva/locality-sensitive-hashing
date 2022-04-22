import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import time
import argparse
import sys as sys
from timeit import default_timer as timer
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import cosine
from scipy import sparse
plt.style.use('ggplot')


def  create_jaccard_signature_matrix(sparse_matrix, permutation_count):
    '''
        Creating the signature matrix for the jaccard similarity 
    '''
    movies_count = sparse_matrix.shape[0] - 1
    users_count = sparse_matrix.shape[1] - 1
    # Initializing the signature matrix with the masx possible int value
    signature_matrix = np.full(shape=(permutation_count, users_count), fill_value=sys.maxsize)

    for i in range(permutation_count):
        # Creating different row permutations
        permuted_rows = np.random.permutation(movies_count)
        # print(permuted_rows)
        permuted_matrix = sparse_matrix[permuted_rows, :]
        # find and save index of first nonzero found in jth column of array:
        # indices to store all the column indices for each of these data
        # indptr[i]:indptr[i+1] to represent the slice in data field to find row[i]'s all non-zero elements
        for j in range(users_count):
            inds = permuted_matrix.indices[permuted_matrix.indptr[j]:permuted_matrix.indptr[j+1]]
            signature_matrix[i, j] = inds.min() if len(inds) > 0 else 0

    return signature_matrix


def create_sparse(data, ones=True):
    '''
        Creating the sparse interaction matrix.

        Only the cosine similarity requires the true values of 
        the ratings. For JS and DCS we use 1s to represent 
        the existence of rating
    '''
    rows = data[:, 1] # movies
    columns = data[:, 0] # users

    if ones:
        interactions = np.ones(data.shape[0])
    else:
        interactions = interactions = list(data[:, 2])
    return sparse.csc_matrix(
                                (interactions, (rows, columns)),
                                dtype='b',
                                shape=(len(np.unique(rows))+1, len(np.unique(columns))+1)
                            )


def parsed_arguments(parser):
    '''
        Parsing the arguements from the command line
    '''
    # parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='path', 
                            help='File path for input data', 
                            default='user_movie_rating.npy', 
                            type=str)
    parser.add_argument('-s', action='store', dest='seed', 
                            help='Chosen seed for run',
                            default=0,
                            type=int)
    parser.add_argument('-m', 
                            action='store', 
                            dest='method', 
                            help='Method for similarity check ("js", "cs", "dcs")',
                            type=str)
    return parser.parse_args()



def signature_similarity(user1, user2, signature_matrix):
    '''
        A method to approximate the similarity of two users in the 
        signature matrix in order to restrict our search space 
        later on during the similarity measure (JS, CS or DCS)
    '''
    # Calculating the non-zero values of the two users that are also equal
    sig_sim = float(np.count_nonzero(signature_matrix[:, user1] == signature_matrix[:, user2])) 
    sig_length = len(signature_matrix[:, user1]) # length of the band
    return sig_sim / sig_length


def calc_jaccard_similarity(sparse_matrix, candidate_pairs, filename, threshold = 0.5):
    '''
        Finding true pairs and appending them on a file, based on Jaccard similarity
    '''
    file = open(filename, "w")
    file.close()
    count = 0
    for i, p in enumerate(candidate_pairs):
        u1 = np.array(sparse_matrix[:, p[0]].toarray()).flatten()
        u2 = np.array(sparse_matrix[:, p[1]].toarray()).flatten()
        similarity = 1 - jaccard(u1,u2)
        if similarity > threshold:
            count += 1
            file = open(filename,'a')
            file.write('{0} {1} {2}\n'.format(p[0], p[1], similarity))
            file.close()
    return count


def cosine_similarity(u1, u2):
    '''
        Method to calculate the cosine similarity between 2 vectors/users.
    '''
    return 1-math.degrees(math.acos(1-cosine(u1,u2)))/180

def create_signature_matrix(sparse_matrix, num_projections):
    '''
        Method to create the CS and DCS signature matrix
    '''
    movies_count = sparse_matrix.shape[0]
    mu, sigma = 0, 1 # mean and standard deviation
    s = np.array([[np.random.normal(mu, sigma, movies_count)] for i in range(num_projections)])
    vectors = s.reshape(num_projections, movies_count)
    signature_matrix = np.array([sparse_matrix.transpose().dot(vector) for vector in vectors])
    signature_matrix[signature_matrix <= 0] = -1
    signature_matrix[signature_matrix > 0] = 1
    return signature_matrix

def calc_cosine_similarity(sparse_matrix, candidate_pairs, filename, cs_threshold = 0.73):
    '''
        Finding true pairs and appending them on a file, based on CS similarity
    '''
    file = open(filename, "w")
    file.close()
    count = 0
    for i, p in enumerate(candidate_pairs):
        u1 = np.array(sparse_matrix[:, p[0]].toarray()).flatten()
        u2 = np.array(sparse_matrix[:, p[1]].toarray()).flatten()
        similarity = cosine_similarity(u1, u2)
        if similarity >= cs_threshold:
            count += 1
            file = open(filename,'a')
            file.write('{0} {1} {2}\n'.format(p[0], p[1], similarity))
            file.close()
    return count


def find_candidate_pairs(signature_matrix, bands_count, sig_sim_thres=0.5):
    '''
        Finding the candidate pairs
    '''
    candidate_pairs = set() # initialise candidate pairs
    rows = math.floor(signature_matrix.shape[0]/bands_count)*bands_count
    # Splitting signature matrix into bands_count times rowwise
    split_sm = np.split(signature_matrix[:rows], bands_count) 
    for n in range(bands_count):
        curr_band = split_sm[n]
        curr_user_index = 0 
        hash_vals = {}
        for curr_usr_idx in range(signature_matrix.shape[1]):
            curr_hash = hash(tuple(curr_band[:,curr_usr_idx]))
            hash_vals.setdefault(curr_hash, []).append(curr_user_index)
            curr_user_index += 1
    
        # add unique candidate pairs with all possible combos in current bucket
        for curr_key in hash_vals:
            if len(hash_vals[curr_key]) > 1 and len(hash_vals[curr_key]) < 150: 
                curr_list = hash_vals[curr_key]
                u_pairs = set(pair for pair in itertools.combinations(curr_list, 2))
                u_pairs = u_pairs.difference(candidate_pairs)
                for u in u_pairs:
                    if signature_similarity(u[0], u[1], signature_matrix) > sig_sim_thres:
                        candidate_pairs.add(u)
    
    return candidate_pairs

def initiate_js(data):
    '''
        Method to initialize the JS measure algorithm to find similar pairs 
        of users
    '''
    print('Initiating the JS algorithm')
    # Creating a sparse csc matrix
    tic = time.perf_counter()
    sparse_matrix  = create_sparse(data)
    # print(sparse_matrix)
    toc = time.perf_counter()
    print(f"Converted to Sparse Matrix in {toc - tic:0.4f} seconds")

    # Creating the signature matrix
    tic = time.perf_counter()
    signature_matrix = create_jaccard_signature_matrix(sparse_matrix, 120)
    toc = time.perf_counter()
    print(f"Created and filled signature matrix {toc - tic:0.4f} seconds")
    # print(signature_matrix)

    # Finding candidate pairs
    tic = time.perf_counter()
    candidate_pairs = find_candidate_pairs(signature_matrix, 21, sig_sim_thres=0.5)
    toc = time.perf_counter()
    print("Found ", len(candidate_pairs), f" candidate pairs in {toc - tic:0.4f} seconds")

    # Finding true pairs from candidate
    tic = time.perf_counter()
    count = calc_jaccard_similarity(sparse_matrix, candidate_pairs, 'js.txt', 0.5)
    toc = time.perf_counter()
    print("Found ", count, f" 'true' pairs in {toc - tic:0.4f} seconds")

def initiate_cs(data):
    '''
        Method to initialize the CS measure algorithm to find similar pairs 
        of users
    '''
    # Creating a sparse csc matrix
    tic = time.perf_counter()
    sparse_matrix  = create_sparse(data, ones=False)
    # print(sparse_matrix)
    toc = time.perf_counter()
    print(f"Converted to Sparse Matrix in {toc - tic:0.4f} seconds")

    # Creating the signature matrix
    tic = time.perf_counter()
    signature_matrix = create_signature_matrix(sparse_matrix, 150) # 200 
    # print(signature_matrix)
    toc = time.perf_counter()
    print(f"Created and filled signature matrix {toc - tic:0.4f} seconds")
    # print(signature_matrix)

    # Finding candidate pairs
    tic = time.perf_counter()
    candidate_pairs = find_candidate_pairs(signature_matrix, 10, sig_sim_thres=0.73) # 17
    toc = time.perf_counter()
    print("Found ", len(candidate_pairs), f" candidate pairs in {toc - tic:0.4f} seconds")

    # Finding true pairs from candidate
    tic = time.perf_counter()
    count = calc_cosine_similarity(sparse_matrix, candidate_pairs, 'cs.txt', 0.73)
    toc = time.perf_counter()
    print("Found ", count, f" 'true' pairs in {toc - tic:0.4f} seconds")


def initiate_dcs(data):
    '''
        Method to initialize the DCS measure algorithm to find similar pairs 
        of users
    '''
    # Creating a sparse csc matrix
    tic = time.perf_counter()
    sparse_matrix  = create_sparse(data, ones=True)
    # print(sparse_matrix)
    toc = time.perf_counter()
    print(f"Converted to Sparse Matrix in {toc - tic:0.4f} seconds")

    # Creating the signature matrix
    tic = time.perf_counter()
    signature_matrix = create_signature_matrix(sparse_matrix, 150) # 200 
    # print(signature_matrix)
    toc = time.perf_counter()
    print(f"Created and filled signature matrix {toc - tic:0.4f} seconds")
    # print(signature_matrix)

    # Finding candidate pairs
    tic = time.perf_counter()
    candidate_pairs = find_candidate_pairs(signature_matrix, 10, sig_sim_thres=0.73) # 17
    toc = time.perf_counter()
    print("Found ", len(candidate_pairs), f" candidate pairs in {toc - tic:0.4f} seconds")

    # Finding true pairs from candidate
    tic = time.perf_counter()
    count = calc_cosine_similarity(sparse_matrix, candidate_pairs, 'dcs.txt', 0.73)
    toc = time.perf_counter()
    print("Found ", count, f" 'true' pairs in {toc - tic:0.4f} seconds")

'''
    Start script
'''
if __name__ == "__main__":
    code_started_at = timer()
    parser = argparse.ArgumentParser()

    args = parsed_arguments(parser)
    if len(sys.argv)!=3:
        parser.print_help(sys.stderr)
        sys.exit(1)
    print("arguments passed:", args)

    np.random.seed(args.seed)
    # Loading the data from the numpy array
    data = np.load(args.path)

    if args.method == 'js':
        initiate_js(data)
    elif args.method == 'cs':
        initiate_cs(data)
    elif args.method == 'dcs':
        initiate_dcs(data)
    else:
        print('Measure method arguement provided was incorrect')
        exit
 
    # Code Ended
    code_ended_at = timer()
    execution_time = np.round((code_ended_at - code_started_at) / 60, decimals = 5)
    print(f'Execution time = {execution_time} minutes')
    print('______________________________________________')