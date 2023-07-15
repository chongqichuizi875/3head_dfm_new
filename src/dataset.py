from torch.utils.data import Dataset
from utils import *
import os
from numpy import int8
import numpy as np
from scipy import sparse as sps
from model import MF, MF_IPS, MF_SNIPS, MF_DR, MF_DR_JL
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import copy
SECONDS_A_DAY = 60 * 60 * 24
SECONDS_AN_HOUR = 60 * 60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY * 5
NUMERICAL_EMB_SIZE = 1488
CATEGORICAL_EMB_SIZE = 2000
num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)

def linear_regression(x, input_dim, output_dim):
    # x: numpy

    x_tensor = torch.from_numpy(x).to(DEVICE).float()

    bn = torch.nn.BatchNorm1d(input_dim).to(DEVICE)
    p_linear = torch.nn.Linear(input_dim, output_dim).to(DEVICE)
    x_tensor = bn(x_tensor)
    p = p_linear(x_tensor)
    p = torch.sigmoid(p).cpu().detach().numpy().reshape(-1)
    return p


def one_hot_encoding_sparse_np(indices, num_classes):
    row_index = np.arange(indices.shape[0])
    data = np.ones(indices.shape[0])
    one_hot_sparse = csr_matrix((data, (row_index, indices)), shape=(indices.shape[0], num_classes))
    return one_hot_sparse

def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row, col]
    x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
    return x, y

def generate_yahoo_or_coat(x_train, y_train, x_test, y_test, num_user, num_item, non_delay_prop=0.6):
    x_train, y_train = shuffle(x_train, y_train)
    y_train = binarize(y_train)
    y_test = binarize(y_test)

    y_train_sparse = sps.csr_matrix((y_train, (x_train[:, 0], x_train[:, 1])), shape=(num_user, num_item),
                                    dtype=np.float32)  # (290, 300)
    y_train_sparse = y_train_sparse.toarray().reshape(num_user * num_item)  # (87000, 1)
    O_train_sparse = sps.csr_matrix((np.ones(len(x_train[:, 0])), (x_train[:, 0], x_train[:, 1])),
                                    shape=(num_user, num_item), dtype=np.float32)
    O_train_sparse = O_train_sparse.toarray().reshape(num_user * num_item)
    print(f"o=1: {len(O_train_sparse[O_train_sparse==1])}")
    print(f"y=1: {len(y_train_sparse[y_train_sparse==1])}")
    "MF"
    embedding_k = 8
    mf = MF(num_user, num_item, embedding_k=embedding_k)
    mf.to(DEVICE)
    hyper = {"num_epoch": 5000, "batch_size": 512, "lr": 1e-2, "lamb": 1e-4, "tol": 1e-5, "verbose": 1, "G": 1}
    # mf.fit(x_train, y_train, hyper, y_ips=O_train_sparse)
    mf.fit(x_train, y_train, hyper, y_ips=None)
    sample = generate_meshgrid(num_user, num_item)  # (87000, 2)
    _, emb_train = mf.predict(torch.from_numpy(sample))  # (87000, 1), (87000, 16)
    _, emb_test = mf.predict(torch.from_numpy(x_test))
    sigma_h = 1
    W_d = np.random.normal(0, sigma_h, 2 * embedding_k)  # (16,)
    lbd = np.exp(np.dot(emb_train, W_d))  # (87000,)
    D = np.random.exponential(lbd)  # (87000,)

    L = np.quantile(D[y_train_sparse == 1], non_delay_prop)  # (1,) get the 60% value
    ts_click = np.random.uniform(0, L, num_user * num_item)  # (87000,)
    E = L - ts_click  # (87000,)
    idx = (D <= E)
    y_train = y_train_sparse * idx
    print(sum(y_train))
    # y_train_sparse[O_train_sparse == 0] = -1
    E[y_train == 1] = D[y_train == 1]
    train_labels = np.c_[np.c_[np.c_[y_train_sparse, D], y_train], E]
    print(np.sum(train_labels, axis=0))
    return {"train": {"x": pd.DataFrame(x_train),
                      "labels": train_labels,
                      "observe": O_train_sparse},
            "test": {"x": pd.DataFrame(x_test),
                     "labels": y_test,
                     }}

def get_criteo_tensor(data_set_dir):
    df = pd.read_csv(data_set_dir, sep="\t", header=None)
    df = df.sample(n=120000, random_state=17)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()
    df = df[df.columns[2:]]
    df.iloc[:, 8:] = df.iloc[:, 8:].fillna("").astype(str)
    df.iloc[:, :8] = df.iloc[:, :8].fillna(-1).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df.rename(columns={col: str(col - 2) for col in df.columns}, inplace=True)
    num_samples = len(df)
    print(f"total sample num: {num_samples}")
    return df, click_ts, pay_ts, num_samples

def preprocessing(params):
    data_dir = "data"
    name = params["data"]
    non_delay_prop = params["D"]
    dataset_dir = os.path.join(data_dir, name)
    if name == "coat":
        train_file = os.path.join(dataset_dir, "train.ascii")
        test_file = os.path.join(dataset_dir, "test.ascii")
        with open(train_file, "r") as f:
            train_mat = []
            for line in f.readlines():
                train_mat.append(line.split())
            train_mat = np.array(train_mat).astype(int)
        with open(test_file, "r") as f:
            test_mat = []
            for line in f.readlines():
                test_mat.append(line.split())
            test_mat = np.array(test_mat).astype(int)
        print(f"===>Load from {name} data set<===")
        print("[train] rating ratio: {:.6f}".format((train_mat > 0).sum() / (train_mat.shape[0] * train_mat.shape[1])))
        print("[test]  rating ratio: {:.6f}".format((test_mat > 0).sum() / (test_mat.shape[0] * test_mat.shape[1])))
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]
        return generate_yahoo_or_coat(x_train, y_train, x_test, y_test, num_user, num_item, non_delay_prop)

    elif name == "yahoo":
        train_file = os.path.join(dataset_dir, "train.txt")
        test_file = os.path.join(dataset_dir, "test.txt")
        train_mat = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                train_mat.append(line.strip().split())
        train_mat = np.array(train_mat).astype(int)
        test_mat = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                test_mat.append(line.strip().split())

        test_mat = np.array(test_mat).astype(int)
        print("===>Load from {} data set<===".format(name))
        print("[train] num data:", train_mat.shape[0])
        print("[test]  num data:", test_mat.shape[0])
        num_user = train_mat[:, 0].max()
        num_item = train_mat[:, 1].max()
        x_train, y_train, x_test, y_test = train_mat[:, :-1], train_mat[:, -1], \
                                           test_mat[:, :-1], test_mat[:, -1]
        x_train = x_train - 1
        x_test = x_test - 1
        return generate_yahoo_or_coat(x_train, y_train, x_test, y_test, num_user, num_item, non_delay_prop)

    elif name == "criteo":
        dataset_dir = os.path.join(dataset_dir, "data.txt")
        df, click_ts, pay_ts, num_samples = get_criteo_tensor(dataset_dir)
        data = DataDF(df, click_ts, pay_ts)
        x = data.x.values
        numerical_emb_concat = None
        for i in range(0, 8):
            feat = x[:, i]
            # Scale feature values to the range [0, num_bin_size[i]-1]
            scaled_feat = (feat * (num_bin_size[i] - 1)).astype(int)
            dummy_matrix = one_hot_encoding_sparse_np(scaled_feat, num_bin_size[i])
            if numerical_emb_concat is None:
                numerical_emb_concat = dummy_matrix
            else:
                numerical_emb_concat = hstack((numerical_emb_concat, dummy_matrix))

        # Concatenate one-hot encoded tensors along the second dimension (columns)
        encoder = LabelEncoder()
        for i in range(8, 17):
            feat = x[:, i]
            feat = encoder.fit_transform(feat)
            scaled_feat = ((feat - feat.min()) / (feat.max() - feat.min()) * (cate_bin_size[i - 8] - 1)).astype(int)
            dummy_matrix = one_hot_encoding_sparse_np(scaled_feat, cate_bin_size[i - 8])
            if numerical_emb_concat is None:
                numerical_emb_concat = dummy_matrix
            else:
                numerical_emb_concat = hstack((numerical_emb_concat, dummy_matrix))
        x = numerical_emb_concat.astype(int8).toarray()
        print(f"x.shape: {x.shape}")
        p_list = np.array([])
        all_index = np.arange(num_samples)
        total_batch = 100
        batch_size = num_samples // total_batch
        for i in range(total_batch):
            torch.cuda.empty_cache()
            print("\r%s" % i, end="")
            if i == total_batch - 1:
                index = all_index[i * batch_size:]
            else:
                index = all_index[i * batch_size: (i + 1) * batch_size]

            p = linear_regression(x[index], CATEGORICAL_EMB_SIZE + NUMERICAL_EMB_SIZE, 1)
            p_list = np.concatenate((p_list, p))

        print()
        observe = np.random.binomial(size=num_samples, n=1, p=p_list)
        print("percentile for observed data: ", sum(observe) / num_samples)
        data.observe = observe
        data.x = pd.DataFrame(x)
        train_data = data.sub_days(0, 30).shuffle()  # 取前30天点击的用户作为训练集
        train_data.pay_ts[train_data.pay_ts < 0] = SECONDS_A_DAY * 30
        delay = np.reshape(train_data.pay_ts - train_data.click_ts, (-1, 1)) / SECONDS_DELAY_NORM
        elapse = np.reshape(SECONDS_A_DAY * 30 - train_data.click_ts, (-1, 1))
        delayed_pay = delay > elapse
        delayed_pay = delayed_pay.reshape(-1)

        y_train = train_data.labels
        y_train[delayed_pay] = 0  # if delay > elapse, then y=0
        y_train[np.where(train_data.observe == 0)] = 0  # if user never observes the item, then y=0
        y_train = np.reshape(y_train, (-1, 1))

        train_data.labels = np.reshape(train_data.labels, (-1, 1))
        train_data.labels = np.concatenate(
            [train_data.labels, delay, y_train, elapse], axis=1)
        test_data = data.sub_days(30, 60)

        return {"train": {"x": train_data.x,
                          "click_ts": train_data.click_ts,
                          "pay_ts": train_data.pay_ts,
                          "sample_ts": train_data.sample_ts,
                          "labels": train_data.labels,
                          "observe": train_data.observe},
                "test": {"x": test_data.x,
                         "click_ts": test_data.click_ts,
                         "pay_ts": test_data.pay_ts,
                         "sample_ts": test_data.sample_ts,
                         "labels": test_data.labels,
                         "observe": test_data.observe}}

    print("Cant find the data set", name)
    return

class DataDF():
    def __init__(self, features, click_ts, pay_ts, observe=None, sample_ts=None, labels=None, delay_label=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.delay_label = delay_label
        self.observe = observe

        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = (pay_ts > 0).astype(np.int32)  # 只要购买, label就为1。这里的label就是C

    def sub_days(self, start_day, end_day):
        start_ts = start_day * SECONDS_A_DAY
        end_ts = end_day * SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(features=self.x.iloc[mask],
                      click_ts=self.click_ts[mask],
                      pay_ts=self.pay_ts[mask],
                      observe=self.observe[mask],
                      sample_ts=self.sample_ts[mask],
                      labels=self.labels[mask])

    def sub_hours(self, start_hour, end_hour):
        start_ts = start_hour * SECONDS_AN_HOUR
        end_ts = end_hour * SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def add_fake_neg(self):
        pos_mask = self.pay_ts > 0
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def only_pos(self):
        pos_mask = self.pay_ts > 0
        print(np.mean(pos_mask))
        print(self.pay_ts[pos_mask].shape)
        return DataDF(self.x.iloc[pos_mask],
                      self.click_ts[pos_mask],
                      self.pay_ts[pos_mask],
                      self.sample_ts[pos_mask],
                      self.labels[pos_mask])

    def to_tn(self):
        mask = np.logical_or(self.pay_ts < 0, self.pay_ts -
                             self.click_ts > SECONDS_AN_HOUR)
        x = self.x.iloc[mask]
        sample_ts = self.sample_ts[mask]
        click_ts = self.click_ts[mask]
        pay_ts = self.pay_ts[mask]
        label = pay_ts < 0
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_dp(self):
        x = self.x
        sample_ts = self.sample_ts
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        label = pay_ts - click_ts > SECONDS_AN_HOUR
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def add_esdfm_cut_fake_neg(self, cut_size):
        mask = self.pay_ts - self.click_ts > cut_size
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts + cut_size, self.pay_ts[mask]],
            axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def to_fsiw_1(self, cd, T):  # build pre-training dataset 1 of FSIW
        mask = np.logical_and(self.click_ts < T - cd, self.pay_ts > 0)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.click_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < T - cd] = 1
        # FSIW needs elapsed time information
        x.insert(x.shape[1], column="elapse", value=(
                                                            T - click_ts - cd) / SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_0(self, cd, T):  # build pre-training dataset 0 of FSIW
        mask = np.logical_or(self.pay_ts >= T - cd, self.pay_ts < 0)
        mask = np.logical_and(self.click_ts < T - cd, mask)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.sample_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < 0] = 1
        x.insert(x.shape[1], column="elapse", value=(
                                                            T - click_ts - cd) / SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        self.x.insert(self.x.shape[1], column="elapse", value=(
                                                                      cut_ts - self.click_ts) / SECONDS_FSIW_NORM)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      label)

    def to_dfm_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        delay = np.reshape(cut_ts - self.click_ts, (-1, 1)) / SECONDS_DELAY_NORM
        labels = np.concatenate([np.reshape(label, (-1, 1)), delay], axis=1)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      labels)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(features=self.x.iloc[idx],
                      click_ts=self.click_ts[idx],
                      pay_ts=self.pay_ts[idx],
                      observe=self.observe[idx],
                      sample_ts=self.sample_ts[idx],
                      labels=self.labels[idx])