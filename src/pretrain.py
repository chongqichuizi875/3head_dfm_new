import json
import os.path
from json import JSONDecodeError

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from dataset import load_data
from models import get_model
import const
import matplotlib.pyplot as plt

# Disable gradient computation messages
torch.set_grad_enabled(False)

# assign the GPU
physical_devices = torch.cuda.device_count()

# now only use single GPU
torch.cuda.set_device(0)
torch.cuda.empty_cache()


def test(model, x_test, y_test, is_Training=False):
    all_logits = []
    all_probs = np.array([])
    all_labels = np.array([])

    # x_test = torch.Tensor(x_test).cuda(const.CUDA_DEVICE)
    num_sample = len(x_test)
    all_idx = np.arange(num_sample)
    batch_size = const.BATCH_SIZE
    total_batch = num_sample // batch_size
    if batch_size > num_sample:
        batch_size = num_sample
        total_batch = 0

    for i in range(total_batch+1):
        if i == total_batch:
            selected_idx = all_idx[batch_size * i:]
        else:
            selected_idx = all_idx[batch_size * i:(i + 1) * batch_size]
        batch_x = x_test[selected_idx]
        batch_x = torch.Tensor(batch_x).cuda(const.CUDA_DEVICE)
        pred = model.forward(batch_x)["C"]
        # pred = model(batch_x, training=False)
        pred_prob = torch.sigmoid(pred).cpu().detach().numpy()
        # all_logits.append(logits)
        all_probs = np.concatenate((all_probs, pred_prob))
        if is_Training:
            all_labels = np.concatenate((all_labels, y_test[selected_idx][:, 0]))
        else:
            all_labels = np.concatenate((all_labels, y_test[selected_idx]))

    print(f"test set shape: {np.array(all_labels).shape}")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    # roc_auc = roc_auc_score(all_labels, all_probs)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    # fig, ax = plt.subplots()
    # ax.plot(recall, precision, label='PR curve (AUC = {:.4f})'.format(pr_auc))
    # ax.plot(fpr, tpr, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.0])
    # ax.set_xlabel('Recall (True Positive Rate)')
    # ax.set_ylabel('Precision (Positive Predictive Value)')
    # ax.set_title('Precision-Recall and ROC Curves')
    # ax.legend(loc='lower left')

    group = {"data": model.name,
             "loss": model.loss.__name__,
             "fpr": fpr.tolist(),
             "tpr": tpr.tolist(),
             "recall": recall.tolist(),
             "precision": precision.tolist(),
             "pr_auc": pr_auc,
             "roc_auc": roc_auc}
    with open("results.json", 'a') as f:
        pass
    with open("results.json", 'r') as f:
        try:
            list1 = json.load(f)
        except JSONDecodeError:
            list1 = []
    append_label = 1
    with open("results.json", 'w') as f:
        if len(list1) == 0:
            list1 = [group]
        else:
            for index, g in enumerate(list1):
                if group["loss"] == g["loss"] and group["data"] == g["data"]:
                    list1[index] = group
                    append_label = 0
                    break

            if append_label == 1:
                list1.append(group)
        json_objects = json.dumps(list1, indent=4)
        f.write(json_objects)

    return roc_auc, pr_auc


# def optim_step(model, x, targets, optimizer, loss_fn, params):
#     optimizer.zero_grad()  # Reset gradients to zero
#     outputs = model(x)
#     reg_loss = sum(model.losses())
#     loss_dict = loss_fn(targets, outputs, params)
#     loss = loss_dict["loss"] + reg_loss
#     loss.backward()  # Compute gradients
#     optimizer.step()  # Update weights


# def train(model, optimizer, train_data, params):
#     for step, batch in enumerate(tqdm(train_data), 1):
#         batch_x = batch[0]
#         batch_y = batch[1]
#         targets = {"label": batch_y}
#         optim_step(model, batch_x, targets, optimizer,
#                    get_loss_fn(params["loss"]), params)


def run(params):
    print(params)
    if const.BUILD_DATA == 1:
        if not os.path.exists("datasets/" + params['data']):
            os.makedirs("datasets/" + params['data'])
        dataset = load_data(params)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        x_train = train_dataset['x'],
        x_train = x_train[0].values
        np.save("datasets/" + params['data'] + "/x_train.npy", x_train)
        x_test = test_dataset['x'],
        x_test = x_test[0].values
        np.save("datasets/" + params['data'] + "/x_test.npy", x_test)
        y_train = train_dataset['labels']
        np.save("datasets/" + params['data'] + "/y_train.npy", y_train)
        y_test = test_dataset['labels']
        np.save("datasets/" + params['data'] + "/y_test.npy", y_test)
    else:
        x_train = np.load("datasets/" + params['data'] + "/x_train.npy", allow_pickle=True)
        x_test = np.load("datasets/" + params['data'] + "/x_test.npy", allow_pickle=True)
        y_train = np.load("datasets/" + params['data'] + "/y_train.npy", allow_pickle=True)
        y_test = np.load("datasets/" + params['data'] + "/y_test.npy", allow_pickle=True)

    # train_data = TensorDataset(
    #     torch.tensor(x_train), torch.tensor(y_train))
    # train_data = DataLoader(
    #     train_data,
    #     batch_size=params["batch_size"],
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=True
    # )
    # test_data = TensorDataset(
    #     torch.tensor(x_test), torch.tensor(y_test)
    # )

    model = get_model(params["model"], params).cuda(const.CUDA_DEVICE)
    model.fit(x_train, y_train, batch_size=1048576)
    # train_auc = test(model, x_train, y_train, is_Training=True)
    # print("training auc: ", train_auc)
    roc_auc, pr_auc = test(model, x_test, y_test)
    print(f"AUC-ROC: {roc_auc}, PR-AUC: {pr_auc}")
    # plt.show()
