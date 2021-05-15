import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import sys
from knockknock import slack_sender

os_path = os.path.realpath("")
random_seed = 777
sys.path.append("./")
from models.model import ATMNet
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ATMNet().to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(target, label, batch_size_num=32, epochs=5):
    train_dataset = TensorDataset(torch.FloatTensor(target), torch.LongTensor(label).squeeze())
    train_loader = DataLoader(train_dataset, batch_size=batch_size_num, shuffle=True, drop_last=True)
    file = open(os_path+"\\log\\model_log.txt", 'w')
    model.train()
    print("Training start.")
    check_loss = math.inf
    for epoch in range(epochs):
        outputs = []
        if epoch == 0:
            print("-" * 100)
        print("Epoch : {0}".format(epoch+1))
        print("-" * 100)
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()    # calc gradients
            optimizer.step()   # update gradients

            if loss.item() < check_loss:
                torch.save(model.state_dict(), os_path+"\\checkpoint\\predictor.pth")
                check_loss = loss.item() # save optimized loss model
                print("Save Model. Iteration : {0}, Loss : {1}".format(idx, loss.item()))
                log = "Save Model. "+"Epoch : "+str(epoch+1)+", Iteration : "+str(idx)+", Loss : "+str(loss.item())+"\n"
                file.write(log)
        print("-" * 100)

    print("Training finish.")

    return {'best model loss': check_loss}

webhook_url = ""
@slack_sender(webhook_url=webhook_url, channel="")
def train_model_slack_notify(target, label, batch_size_num, epochs):
    return train(target, label, batch_size_num, epochs)
