import os
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import torch.nn as nn

def load_data_util(dataset):
    if dataset == 'german':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german('german', label_number=100, path=os.path.join("dataset", "german"))
    elif dataset == 'bail':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail', label_number=100, path=os.path.join("dataset", "bail"))
    elif dataset == "math":
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_math(path=os.path.join("dataset", "math"))
    elif dataset == "por":
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_por(path=os.path.join("dataset", "por"))
    else:
        adj, features, labels, idx_train, idx_val, idx_test, sens = None, None, None, None, None, None, None
        print("This dataset is not supported up to now!")
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def seed_everything(seed=0):
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def normalize_scipy(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map =  np.array(idx_map)

    return idx_map

def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bail/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)


    # sensitive feature removal
    # header.remove('WHITE')

    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    # the label here is the prediction label and the goal of following code is to avoid class (utility prediction) imbalance
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute - Fix pandas SettingWithCopyWarning
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Female', 'Gender'] = 1
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Male', 'Gender'] = 0

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_math(dataset="math", sens_attr="sex", predict_attr="G3", path="dataset/math", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        print("build relationship")
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.9)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_por(dataset="por", sens_attr="sex", predict_attr="G3", path="dataset/por", label_number=10000):
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        print("build relationship")
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.9)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(1)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_data_util_similarity_based(dataset, t1=0.6, t2=0.2, t3=0.2, fair_noise=0.0, path="./dataset/", verbose=True, seed=42):
    """
    New similarity-based dataset splitting method with fair noise
    
    Args:
        dataset: dataset name ('german', 'bail', 'math', 'por')
        t1: training ratio (default: 0.6)
        t2: validation ratio (default: 0.2) 
        t3: testing ratio (default: 0.2)
        fair_noise: probability to ignore prediction label differences (0.0-1.0, default: 0.0)
        path: dataset path
        verbose: whether to print detailed logs
        seed: random seed for reproducibility
    
    Returns:
        adj, features, labels, idx_train, idx_val, idx_test, sens
    """
    assert abs(t1 + t2 + t3 - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    if dataset == 'german':
        return _load_german_similarity_based(t1, t2, t3, fair_noise, path + "german/", verbose, seed)
    elif dataset == 'bail':
        return _load_bail_similarity_based(t1, t2, t3, fair_noise, path + "bail/", verbose, seed)
    elif dataset == "math":
        return _load_math_similarity_based(t1, t2, t3, fair_noise, path + "math/", verbose, seed)
    elif dataset == "por":
        return _load_por_similarity_based(t1, t2, t3, fair_noise, path + "por/", verbose, seed)
    else:
        print("This dataset is not supported up to now!")
        return None, None, None, None, None, None, None

def _load_german_similarity_based(t1, t2, t3, fair_noise, path, verbose, seed):
    """Load German dataset with similarity-based splitting"""
    idx_features_labels = pd.read_csv(os.path.join(path, "german.csv"))
    header = list(idx_features_labels.columns)
    header.remove('GoodCustomer')
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute - Fix pandas SettingWithCopyWarning
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Female', 'Gender'] = 1
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Male', 'Gender'] = 0

    # build relationship
    if os.path.exists(f'{path}/german_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/german_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/german_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels['GoodCustomer'].values
    labels[labels == -1] = 0
    sens = idx_features_labels['Gender'].values.astype(int)

    # Convert to torch tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    # Build adjacency matrix
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Apply similarity-based splitting
    idx_train, idx_val, idx_test = _similarity_based_split(features, labels, sens, t1, t2, t3, fair_noise, verbose, seed)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def _load_bail_similarity_based(t1, t2, t3, fair_noise, path, verbose, seed):
    """Load Bail dataset with similarity-based splitting"""
    idx_features_labels = pd.read_csv(os.path.join(path, "bail.csv"))
    header = list(idx_features_labels.columns)
    header.remove('RECID')

    # build relationship
    if os.path.exists(f'{path}/bail_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/bail_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/bail_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels['RECID'].values
    sens = idx_features_labels['WHITE'].values.astype(int)

    # Convert to torch tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    # Build adjacency matrix
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Apply similarity-based splitting
    idx_train, idx_val, idx_test = _similarity_based_split(features, labels, sens, t1, t2, t3, fair_noise, verbose, seed)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def _load_math_similarity_based(t1, t2, t3, fair_noise, path, verbose, seed):
    """Load Math dataset with similarity-based splitting"""
    idx_features_labels = pd.read_csv(os.path.join(path, "math.csv"))
    header = list(idx_features_labels.columns)
    header.remove('G3')

    # build relationship
    if os.path.exists(f'{path}/math_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/math_edges.txt').astype('int')
    else:
        print("build relationship")
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.9)
        np.savetxt(f'{path}/math_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels['G3'].values
    sens = idx_features_labels['sex'].values.astype(int)

    # Convert to torch tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    # Build adjacency matrix
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Apply similarity-based splitting
    idx_train, idx_val, idx_test = _similarity_based_split(features, labels, sens, t1, t2, t3, fair_noise, verbose, seed)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def _load_por_similarity_based(t1, t2, t3, fair_noise, path, verbose, seed):
    """Load Por dataset with similarity-based splitting"""
    idx_features_labels = pd.read_csv(os.path.join(path, "por.csv"))
    header = list(idx_features_labels.columns)
    header.remove('G3')

    # build relationship
    if os.path.exists(f'{path}/por_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/por_edges.txt').astype('int')
    else:
        print("build relationship")
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.9)
        np.savetxt(f'{path}/por_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels['G3'].values
    sens = idx_features_labels['sex'].values.astype(int)

    # Convert to torch tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    # Build adjacency matrix
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Apply similarity-based splitting
    idx_train, idx_val, idx_test = _similarity_based_split(features, labels, sens, t1, t2, t3, fair_noise, verbose, seed)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens

def _similarity_based_split(features, labels, sens, t1, t2, t3, fair_noise, verbose, seed):
    """
    Core similarity-based splitting algorithm with fair noise
    
    Args:
        features: feature matrix
        labels: prediction labels
        sens: sensitive labels
        t1, t2, t3: split ratios (t1 + t2 + t3 = 1.0)
        fair_noise: probability to randomly pick two samples into testing (0.0-1.0)
        verbose: whether to print detailed logs
        seed: random seed for reproducibility
    
    Returns:
        idx_train, idx_val, idx_test: indices for each split
    """
    if verbose:
        print(f"Applying similarity-based splitting with ratios: Training={t1:.1%}, Validation={t2:.1%}, Testing={t3:.1%}")
        print(f"Fair noise parameter: {fair_noise:.1%}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Step 1: Split data into two groups based on sensitive label
    sens_0_indices = torch.where(sens == 0)[0]  # Group s1
    sens_1_indices = torch.where(sens == 1)[0]  # Group s2
    
    if verbose:
        print(f"Sensitive group 0 (s1): {len(sens_0_indices)} samples")
        print(f"Sensitive group 1 (s2): {len(sens_1_indices)} samples")
    
    # Pre-normalize features once for cosine similarity
    feat_norm = features.float()
    norms = torch.norm(feat_norm, dim=1, keepdim=True) + 1e-8
    feat_norm = feat_norm / norms
    
    # Step 2: Find similar pairs between groups with same prediction labels
    testing_candidates = []
    used_indices = set()
    
    # Calculate total testing capacity needed
    total_samples = len(features)
    testing_capacity = int(total_samples * (t2 + t3))
    
    if verbose:
        print(f"Target testing capacity: {testing_capacity} samples")
    
    # Helper to add random samples per fair noise
    def add_random_samples(num_to_add: int):
        nonlocal testing_candidates, used_indices
        remaining = [i for i in range(total_samples) if i not in used_indices]
        if not remaining or num_to_add <= 0:
            return 0
        k = min(num_to_add, len(remaining))
        samples = random.sample(remaining, k)
        testing_candidates.extend(samples)
        used_indices.update(samples)
        return k
    
    # Prepare label-wise candidate indices in s2 for fast filtering
    s2_label_0 = set(torch.where(labels[sens_1_indices] == 0)[0].cpu().numpy().tolist())
    s2_label_1 = set(torch.where(labels[sens_1_indices] == 1)[0].cpu().numpy().tolist())
    # Map from local index in sens_1_indices to global index
    sens_1_list = sens_1_indices.cpu().numpy().tolist()
    
    # Shuffle s1 order for fairness
    s1_list = sens_0_indices.cpu().numpy().tolist()
    random.shuffle(s1_list)
    
    # Bound the amount of work to avoid long runtimes on large datasets
    max_s1_checks = max(testing_capacity * 5, 1000)
    checks_done = 0
    
    for idx_s1 in s1_list:
        if len(testing_candidates) >= testing_capacity:
            break
        if checks_done >= max_s1_checks:
            break
        checks_done += 1
        
        if idx_s1 in used_indices:
            continue
        
        # With probability fair_noise, randomly select two samples
        remaining_slots = testing_capacity - len(testing_candidates)
        if fair_noise > 0 and random.random() < fair_noise and remaining_slots > 0:
            added = add_random_samples(min(2, remaining_slots))
            if added > 0:
                continue
        
        # Otherwise, find most similar point in s2 with SAME prediction label
        y = labels[idx_s1].item()
        if y == 0:
            candidate_local = list(s2_label_0)
        else:
            candidate_local = list(s2_label_1)
        
        if not candidate_local:
            continue
        
        # Convert local indices to global indices and filter out used
        candidate_global = [sens_1_list[i] for i in candidate_local if sens_1_list[i] not in used_indices]
        if not candidate_global:
            continue
        
        # Compute cosine similarity vectorized: sim = feat_norm[candidate_global] @ feat_norm[idx_s1]
        feat_s1_norm = feat_norm[idx_s1]
        feat_s2_norm = feat_norm[torch.LongTensor(candidate_global)]
        sims = torch.mv(feat_s2_norm, feat_s1_norm)
        best_idx = torch.argmax(sims).item()
        best_sim = sims[best_idx].item()
        best_global = candidate_global[best_idx]
        
        # If we found a similar pair above threshold, add both
        if best_sim > 0.5:
            if len(testing_candidates) + 2 <= testing_capacity:
                testing_candidates.extend([idx_s1, best_global])
                used_indices.add(idx_s1)
                used_indices.add(best_global)
            else:
                if len(testing_candidates) < testing_capacity:
                    testing_candidates.append(idx_s1)
                    used_indices.add(idx_s1)
    
    # If we don't have enough testing candidates, add more samples randomly
    if len(testing_candidates) < testing_capacity:
        remaining_needed = testing_capacity - len(testing_candidates)
        add_random_samples(remaining_needed)
    
    # Step 3: Randomly split testing candidates into validation and test
    random.shuffle(testing_candidates)
    
    val_size = int(len(testing_candidates) * (t2 / (t2 + t3))) if (t2 + t3) > 0 else 0
    idx_val = torch.LongTensor(testing_candidates[:val_size])
    idx_test = torch.LongTensor(testing_candidates[val_size:])
    
    # Step 4: Remaining samples go to training
    remaining_indices = [i for i in range(total_samples) if i not in used_indices]
    idx_train = torch.LongTensor(remaining_indices)
    
    if verbose:
        print(f"Final split sizes:")
        print(f"  Training: {len(idx_train)} samples ({len(idx_train)/total_samples:.1%})")
        print(f"  Validation: {len(idx_val)} samples ({len(idx_val)/total_samples:.1%})")
        print(f"  Testing: {len(idx_test)} samples ({len(idx_test)/total_samples:.1%})")
    
    # Do not print fairness scores here when verbose is False, leave to caller
    if verbose:
        _print_split_fairness_scores(labels, sens, idx_train, idx_val, idx_test)
    
    return idx_train, idx_val, idx_test

def _print_split_fairness_scores(labels, sens, idx_train, idx_val, idx_test):
    """Print fairness scores for each dataset split"""
    print("\n=== Fairness Scores for Each Split ===")
    
    # Training set fairness
    sp_train, eo_train = fair_metric(labels[idx_train].cpu().numpy(), 
                                    labels[idx_train].cpu().numpy(),
                                    sens[idx_train].cpu().numpy())
    fairness_train = (sp_train + eo_train) / 2
    print(f"Training Set: SP={sp_train:.4f}, EO={eo_train:.4f}, Fairness Score={fairness_train:.4f}")
    
    # Validation set fairness
    sp_val, eo_val = fair_metric(labels[idx_val].cpu().numpy(), 
                                labels[idx_val].cpu().numpy(),
                                sens[idx_val].cpu().numpy())
    fairness_val = (sp_val + eo_val) / 2
    print(f"Validation Set: SP={sp_val:.4f}, EO={eo_val:.4f}, Fairness Score={fairness_val:.4f}")
    
    # Testing set fairness
    sp_test, eo_test = fair_metric(labels[idx_test].cpu().numpy(), 
                                  labels[idx_test].cpu().numpy(),
                                  sens[idx_test].cpu().numpy())
    fairness_test = (sp_test + eo_test) / 2
    print(f"Testing Set: SP={sp_test:.4f}, EO={eo_test:.4f}, Fairness Score={fairness_test:.4f}")
    
    print("=====================================\n")

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    print(min_values.shape)
    max_values = features.max(axis=0)[0]
    t = max_values-min_values
    for i in range(len(t)):
        if t[i] == 0:
            t[i] = 1
    norm_features = 2*(features - min_values).div(t) - 1
    # print("after norm: ", norm_features.sum())

    # min_values = features.min(axis=0)[0]
    # max_values = features.max(axis=0)[0]
    # norm_features = 2*(features - min_values).div(max_values-min_values) - 1

    return norm_features

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def SWloss_fast(x, y, L=512):
    r = torch.randn(x.shape[1], L).cuda()
    xt = x @ r # (N, 8)
    yt = y @ r # (N, 8)
    xt = torch.sort(xt, 0)[0]
    yt = torch.sort(yt, 0)[0]
    t = xt - yt # (N, 3)
    SW = (t**2).sum()/L
    return SW

def group_distance(hidden, idx_train, sens, labels):
    # return the embedding distance of two subgroups
    sens_ = sens[idx_train]
    labels_ = labels[idx_train]
    hidden_ = hidden[idx_train]

    # obtain idx (idx_s0_y1) for data whose sensitive label=0 and utility label=1
    idx_s0 = sens_==0
    idx_s1 = sens_==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels_==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels_==1)
    idx_s0_y0 = np.bitwise_and(idx_s0, labels_==0)
    idx_s1_y0 = np.bitwise_and(idx_s1, labels_==0)

    hidden_s0_y1 = hidden_[idx_s0_y1]
    hidden_s1_y1 = hidden_[idx_s1_y1]
    hidden_s0_y0 = hidden_[idx_s0_y0]
    hidden_s1_y0 = hidden_[idx_s1_y0]

    sample_number = int(min(hidden_s0_y1.shape[0], hidden_s1_y1.shape[0])/2)
    random_idx_s0_y1 = random.sample(range(hidden_s0_y1.shape[0]), sample_number)
    random_idx_s1_y1 = random.sample(range(hidden_s1_y1.shape[0]), sample_number)

    sample_number = int(min(hidden_s0_y0.shape[0], hidden_s1_y0.shape[0])/2)
    random_idx_s0_y0 = random.sample(range(hidden_s0_y0.shape[0]), sample_number)
    random_idx_s1_y0 = random.sample(range(hidden_s1_y0.shape[0]), sample_number)

    x1 = hidden_s0_y1[random_idx_s0_y1]
    y1 = hidden_s1_y1[random_idx_s1_y1]
    x0 = hidden_s0_y0[random_idx_s0_y0]
    y0 = hidden_s1_y0[random_idx_s1_y0]

    distance_1 = SWloss_fast(x1, y1)
    distance_0 = SWloss_fast(x0, y0)
    
    return distance_1+distance_0
