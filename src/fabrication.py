import torch
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score
from dataset import load_data, rating_mat_to_sample
from utils import *
from models import MF
from scipy import sparse as sps
import pandas as pd


if __name__ == "__main__":
    seed = 2298839
    dataset_name = "coat"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dataset_name == "coat":
        train_mat, test_mat = load_data(dataset_name)
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]
    else:
        x_train, y_train, x_test, y_test = load_data("yahoo")
        num_user = x_train[:,0].max()
        num_item = x_train[:,1].max()
    print("# user: {}, # item: {}".format(num_user, num_item))
    x_train, y_train = shuffle(x_train, y_train)
    y_train = binarize(y_train)
    y_test = binarize(y_test)
    "MF"
    embedding_k = 8
    mf = MF(num_user, num_item, embedding_k=embedding_k, batch_size=128)
    mf.cuda()
    mf.fit(x_train, y_train,
           lr=0.01,
           lamb=1e-4,
           tol=1e-5,
           verbose=False)
    sample = generate_meshgrid(num_user, num_item)  # (87000, 2)
    _, emb = mf.predict(sample)  # (87000, 16)
    y_train_sparse = sps.csr_matrix((y_train, (x_train[:, 0], x_train[:, 1])), shape=(num_user, num_item),
                                    dtype=np.float32)  # (290, 300)
    y_train_sparse = y_train_sparse.toarray().reshape(num_user * num_item)  # (87000, 1)
    O_train_sparse = sps.csr_matrix((np.ones(len(x_train[:, 0])), (x_train[:, 0], x_train[:, 1])),
                                    shape=(num_user, num_item), dtype=np.float32)
    O_train_sparse = O_train_sparse.toarray().reshape(num_user * num_item)
    sigma_h = 1
    W_d = np.random.normal(0, sigma_h, 2 * embedding_k)  # (16,)
    lbd = np.exp(np.dot(emb, W_d))  # (87000,)
    D = np.random.exponential(lbd)  # (87000,)

    L = np.quantile(D[y_train_sparse == 1], 0.6)  # (1,) get the 60% value
    ts_click = np.random.uniform(0, L, num_user * num_item)  # (87000,)
    E = L - ts_click  # (87000,)
    idx = (D <= E)
    y_train = y_train_sparse * idx
    y_train_sparse[O_train_sparse == 0] = -1
    E[y_train == 1] = D[y_train == 1]
    train_data = np.c_[np.c_[np.c_[np.c_[np.array(sample), O_train_sparse], y_train_sparse], y_train], E]
    train_data = pd.DataFrame(train_data, columns=['Uid', 'Iid', 'O', 'C', 'Y_obs', 'E'])
    train_data.to_csv("../"+dataset_name+".csv", index=False)
