import os
from dataset import *
from model import *
from loss import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import json
from json import JSONDecodeError
import matplotlib.pyplot as plt
import matplotlib
matplotlib.matplotlib_fname()
TEST_BATCH = 8192

def test(model, x_test, y_test, is_Training=False):
    all_probs = np.array([])
    all_labels = np.array([])
    num_sample = len(x_test)
    all_idx = np.arange(num_sample)
    batch_size = TEST_BATCH
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
        batch_x = torch.Tensor(batch_x).to(DEVICE)
        pred = model.predict(batch_x)
        # pred = model(batch_x, training=False)
        pred_prob = pred.cpu().detach().numpy()  
        # all_logits.append(logits)
        all_probs = np.concatenate((all_probs, pred_prob))
        if is_Training:
            all_labels = np.concatenate((all_labels, y_test[selected_idx][:, 0]))
        else:
            all_labels = np.concatenate((all_labels, y_test[selected_idx]))
    print(f"test set shape: {np.array(all_labels).shape}")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

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
    threshold = np.mean(all_probs)
    class_pred = (all_probs > threshold).astype(int)
    
    print(f"max: {max(all_probs)}, min: {min(all_probs)}, mean: {np.mean(all_probs)}, var: {np.var(all_probs)}")
    print('---------------')
    print(f"label=1: {len(all_labels[all_labels==1])}, label=0: {len(all_labels[all_labels==0])}")
    TP = np.sum((class_pred == 1) & (all_labels == 1))
    FP = np.sum((class_pred == 1) & (all_labels == 0))
    FN = np.sum((class_pred == 0) & (all_labels == 1))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return roc_auc, pr_auc, P, R

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # set to train mode
    model.train()
    for epoch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            loss, current = loss, (epoch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def run(params):
    print(f"training params: {params}")
    data = params["data"]
    datapath = os.path.join(params["datapath"], data)
    
    if not os.path.exists(datapath):
            os.makedirs(datapath)
    if params["build_data"]:
        dataset = preprocessing(params)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        x_train = train_dataset['x'],
        x_train = x_train[0].values
        processed_datapath = os.path.join("processed_data", params['data'])
        if not os.path.exists(processed_datapath):
            os.makedirs(processed_datapath)
        
        np.save("processed_data/" + params['data'] + "/x_train.npy", x_train)
        x_test = test_dataset['x'],
        x_test = x_test[0].values
        np.save("processed_data/" + params['data'] + "/x_test.npy", x_test)
        y_train = train_dataset['labels']
        np.save("processed_data/" + params['data'] + "/y_train.npy", y_train)
        y_test = test_dataset['labels']
        np.save("processed_data/" + params['data'] + "/y_test.npy", y_test)

    else:
        x_train = np.load("processed_data/" + params['data'] + "/x_train.npy", allow_pickle=True)
        x_test = np.load("processed_data/" + params['data'] + "/x_test.npy", allow_pickle=True)
        y_train = np.load("processed_data/" + params['data'] + "/y_train.npy", allow_pickle=True)
        y_test = np.load("processed_data/" + params['data'] + "/y_test.npy", allow_pickle=True)
    

    if params["mode"] != "eval":
        model = MLP(params).to(params["device"])
        model.fit(x_train, y_train, 
                  batch_size=params["batch_size"], 
                  num_epoch=params["epoch"], 
                  lr=params["lr"],
                  lamb=params["weight_decay"])
        if params["mode"] == "both":
            roc_auc, pr_auc, P, R = test(model, x_test, y_test)
            print(f"ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}, precision: {P}, recall: {R}")
        if params["save_model"]:
            torch.save(model, os.path.join(params["dump_path"], 'model.pth'))
        
    else:
        model = torch.load(os.path.join(params["dump_path"], 'model.pth'))
        # model.fit(x_train, y_train, batch_size=1048576)
        roc_auc, pr_auc, P, R = test(model, x_test, y_test)
        print(f"ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}, precision: {P}, recall: {R}")
    
    