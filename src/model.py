import numpy as np
import torch

from loss import three_head_loss, es_dfm_loss, default_bce_loss, oracle_loss
from utils import generate_meshgrid, sigmoid, compute_ips, DEVICE

torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
import pdb
lossmap = {
    "dfm": default_bce_loss,
    "esdfm": es_dfm_loss,
    "3head": three_head_loss,
    "ora": oracle_loss,
}
class MLP(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.name = params["data"]
        self.loss = lossmap[params["method"]]
        if self.name == "criteo":
            self.feature_dim = 3488
            self.fc0 = torch.nn.Linear(self.feature_dim, 512)
            self.bn0 = nn.BatchNorm1d(256)
            self.fc1 = torch.nn.Linear(512, 256)
            self.bn1 = nn.BatchNorm1d(256)

            self.fc2 = torch.nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)

            self.fc3 = torch.nn.Linear(128, 3)
            self.fc3_1 = torch.nn.Sequential(torch.nn.Linear(128, 64), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(64),
                                             torch.nn.Linear(64, 1))
            self.fc3_2 = torch.nn.Sequential(torch.nn.Linear(128, 64), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(64),
                                             torch.nn.Linear(64, 1))
            self.fc3_3 = torch.nn.Sequential(torch.nn.Linear(128, 64), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(64),
                                             torch.nn.Linear(64, 1))
            
        else:
            
            self.W = torch.nn.Embedding(290, 8)
            self.H = torch.nn.Embedding(300, 8)
            # self.fc5 = torch.nn.Linear(16, 3)
            self.fc0 = torch.nn.Linear(16, 16)
            self.bn0 = nn.BatchNorm1d(16)
            self.fc1 = torch.nn.Linear(16, 8)
            self.bn1 = nn.BatchNorm1d(8)
            self.fc3 = torch.nn.Linear(8, 3)
            
            self.fc3_1 = torch.nn.Sequential(torch.nn.Linear(8, 4), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(4),
                                             torch.nn.Linear(4, 1))
            self.fc3_2 = torch.nn.Sequential(torch.nn.Linear(8, 4), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(4),
                                             torch.nn.Linear(4, 1))
            self.fc3_3 = torch.nn.Sequential(torch.nn.Linear(8, 4), 
                                             torch.nn.Sigmoid(),
                                             nn.BatchNorm1d(4),
                                             torch.nn.Linear(4, 1))
    

    def forward(self, x):
        # x = x.to('cpu').long()
        # user_idx = torch.LongTensor(x[:, 0]).to(DEVICE)
        # item_idx = torch.LongTensor(x[:, 1]).to(DEVICE)
        # U_emb = self.W(user_idx)
        # V_emb = self.H(item_idx)
        # x = torch.cat([U_emb, V_emb], axis=1)

        # x = self.fc5(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.name == "criteo":
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
        if self.params["hard_share"]:
            # print(min(x[:,0]), max(x[:,0]))
            x = self.fc3(x)
            return {"C": x[:, 0], "lamb": x[:, 1], "observe": x[:, 2]}
        else:
            x1 = self.fc3_1(x).squeeze()
            x2 = self.fc3_1(x).squeeze()
            x3 = self.fc3_1(x).squeeze()
            return {"C": x1, "lamb": x2, "observe": x3}

        
    def fit(self, x, y, num_epoch=100, tol=1e-2, verbose=1, batch_size=1048576, lr=0.01, lamb=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = x.shape[0]
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch + 1):

                # mini-batch training
                if idx == total_batch:
                    selected_idx = all_idx[batch_size * idx:]
                else:
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_x = torch.Tensor(sub_x).to(DEVICE)
                sub_labels = y[selected_idx]
                sub_labels = torch.Tensor(sub_labels).to(DEVICE)
                pred = self.forward(sub_x)

                loss = self.loss(sub_labels, pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 2:
                    print(f"[{self.loss.__name__}] Early stop at epoch:{epoch}, xent:{epoch}")
                    break
                early_stop += 1

            last_loss = epoch_loss

            if verbose:
                print(f"[{self.loss.__name__}] epoch:{epoch}, xent:{epoch_loss}")

            if epoch == num_epoch - 1:
                print(f"[{self.loss.__name__}] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        return self.sigmoid(self.forward(x)["C"])
        


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = x[:, 0].to(DEVICE)
        item_idx = x[:, 1].to(DEVICE)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), dim=1)
        if is_training:
            return out
        else:
            return out, torch.cat((U_emb, V_emb), dim=1)

    def predict(self, x):
        with torch.no_grad():
            # x = torch.tensor(x).to(torch.int)
            x = x.clone().detach()
            pred, emb = self.forward(x)
            pred = self.sigmoid(pred).cpu().numpy()
            return pred, emb.cpu().numpy()

    def fit(self, x, y, hyper, y_ips=None):
        num_epoch = hyper["num_epoch"]
        batch_size = hyper["batch_size"]
        lr = hyper["lr"]
        lamb = hyper["lamb"]
        tol = hyper["tol"]
        verbose = hyper["verbose"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0
        x = torch.from_numpy(x).to(torch.int)
        if batch_size > num_sample:
            batch_size = num_sample
            total_batch = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch + 1):
                # mini-batch training
                if idx == total_batch:
                    selected_idx = all_idx[batch_size * idx:]
                else:
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(DEVICE)

                pred = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                # loss = loss_func(sub_y, pred, one_over_zl, selected_idx)
                loss = self.xent_func(pred, sub_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().cpu().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print(f"[MF] epoch:{epoch}, xent:{epoch_loss}")
                    break
                early_stop += 1

            last_loss = epoch_loss
            if epoch % 10 == 0 and verbose:
                print(f"[MF] epoch:{epoch}, xent:{epoch_loss}")

            if epoch == num_epoch - 1:
                print(f"[MF] Reach preset epochs, it seems does not converge.")


class MF_IPS(MF):
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__(num_users, num_items, embedding_k)


class MF_SNIPS(MF):
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__(num_users, num_items, embedding_k)


class MF_DR(MF):
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__(num_users, num_items, embedding_k)

    def fit(self, x, y, hyper, y_ips=None):
        num_epoch = hyper["num_epoch"]
        batch_size = hyper["batch_size"]
        lr = hyper["lr"]
        lamb = hyper["lamb"]
        tol = hyper["tol"]
        verbose = hyper["verbose"]
        G = hyper["G"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_meshgrid(self.num_users, self.num_items)

        num_sample = len(x)  # 6960
        total_batch = num_sample // batch_size
        one_over_zl = compute_ips(x, y, y_ips).to(DEVICE)
        if y_ips is None:
            prior_y = 0
        else:
            prior_y = y_ips.mean()
        early_stop = 0
        if batch_size > num_sample:
            batch_size = num_sample
            total_batch = 0
        x = torch.from_numpy(x).to(torch.int)
        x_all = torch.from_numpy(x_all).to(torch.int)
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)  # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])  # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch + 1):
                # mini-batch training
                if idx == total_batch:
                    selected_idx = all_idx[batch_size * idx:]
                    last_batch = num_sample - idx * batch_size
                    x_sampled = x_all[ul_idxs[G * idx * batch_size:G * idx * batch_size + last_batch]]
                    imputation_y = torch.Tensor([prior_y] * G * last_batch).to(DEVICE)
                else:
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                    x_sampled = x_all[ul_idxs[G * idx * batch_size: G * (idx + 1) * batch_size]]
                    imputation_y = torch.Tensor([prior_y] * G * batch_size).to(DEVICE)
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                # propensity score
                inv_prop = one_over_zl[selected_idx].to(DEVICE)

                sub_y = torch.Tensor(sub_y).to(DEVICE)

                pred = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                pred_ul = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="mean")  # o*eui/pui

                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="mean")  # e^ui

                ips_loss = xent_loss - imputation_loss
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")


class MF_DR_JL(MF):
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__(num_users, num_items, embedding_k)
        self.prediction_model = MF(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)

    def predict(self, x):
        pred, emb = self.prediction_model.predict(x)
        pred = self.sigmoid(torch.tensor(pred)).detach().cpu().numpy()
        return pred, emb

    def fit(self, x, y, hyper, y_ips=None):
        num_epoch = hyper["num_epoch"]
        batch_size = hyper["batch_size"]
        lr = hyper["lr"]
        lamb = hyper["lamb"]
        tol = hyper["tol"]
        verbose = hyper["verbose"]
        G = hyper["G"]
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        # generate all counterfactuals and factuals
        x_all = generate_meshgrid(self.num_users, self.num_items)

        num_sample = len(x)  # 6960
        total_batch = num_sample // batch_size
        one_over_zl = compute_ips(x, y, y_ips)
        early_stop = 0
        if batch_size > num_sample:
            batch_size = num_sample
            total_batch = 0
        x = torch.from_numpy(x).to(torch.int)
        x_all = torch.from_numpy(x_all).to(torch.int)
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)  # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])  # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch + 1):
                # mini-batch training
                if idx == total_batch:
                    selected_idx = all_idx[batch_size * idx:]
                    last_batch = num_sample - idx * batch_size
                    x_sampled = x_all[ul_idxs[G * idx * batch_size:G * idx * batch_size + last_batch]]
                else:
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                    x_sampled = x_all[ul_idxs[G * idx * batch_size: G * (idx + 1) * batch_size]]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].to(DEVICE)

                sub_y = torch.Tensor(sub_y).to(DEVICE)
                pred, _ = self.prediction_model.forward(sub_x)
                imputation_y, _ = self.imputation.predict(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(torch.tensor(imputation_y)).to(DEVICE)

                pred_u, _ = self.prediction_model.forward(x_sampled)
                imputation_y1, _ = self.imputation.predict(x_sampled)
                pred_u = self.sigmoid(pred_u)
                imputation_y1 = self.sigmoid(torch.tensor(imputation_y1)).to(DEVICE)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="mean")  # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="mean")

                ips_loss = xent_loss - imputation_loss
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="mean")
                direct_loss = direct_loss

                loss = ips_loss + direct_loss
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

                pred, _ = self.prediction_model.predict(sub_x)
                imputation_y, _ = self.imputation.forward(sub_x)
                pred = self.sigmoid(torch.tensor(pred)).to(DEVICE)
                imputation_y = self.sigmoid(imputation_y)
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).mean()
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")