import numpy as np 
import argparse
import os 
import random 
import datetime
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from math import ceil

from mini_torch.tensor import Tensor
from mini_torch.layer import DenseLayer
from mini_torch.initializer import XavierInitializer
from mini_torch.activation import ReLU
from mini_torch.network import Network
from mini_torch.model import Model
from mini_torch.loss import SquareLoss
from mini_torch.optimizer import SGD
from mini_torch.data_loader import DataLoader

from utils import draw_scatter, load_data

def train(args, model, features, labels, dev_dataset=None):
    train_x, train_y = Tensor(features), Tensor(labels)

    if dev_dataset is not None:
        dev_x_numpy, dev_y_numpy = dev_dataset[:, :-1], dev_dataset[:, -1:]
        dev_x, dev_y = Tensor(dev_x_numpy), Tensor(dev_y_numpy)
    
    if args.max_steps > 0:
        args.train_epoch_num = ceil(args.max_steps / args.train_batch_size)

    dataloader = DataLoader(train_x, train_y, args.train_batch_size)
    
    print("*********** Train ***********")
    print("\tExamples num: %d"%(dataloader.len))
    print("\tEpoch num: %d"%(args.train_epoch_num))
    print("\tBatch size: %d"%(args.train_batch_size))
    print("\tBatch num: %d"%(dataloader.len))
    
    fig = plt.figure()
    ax3 = Axes3D(fig)
    plt.ion()
    plt.show()

    global_steps = 0
    for epoch in range(args.train_epoch_num):
        epoch_loss = 0
        for batch in dataloader():
            model.zero_grad()
            preds = model.forward(batch.inputs)
            loss = model.loss_layer.loss(preds, batch.labels)
            epoch_loss += loss.values

            loss.backward()
            model.step()
            global_steps += 1

            if global_steps == args.max_steps:
                break
        if dev_dataset is not None and global_steps % 5 == 0:
            dev_preds = model.forward(dev_x)

            plt.cla()
            visualize(ax3, dev_x_numpy, dev_y_numpy, dev_preds.values)
            plt.pause(0.1)

            dev_loss = model.loss_layer.loss(dev_preds, dev_y)
            print("****Draw on epoch-%d, dev loss: %f****"%(epoch, dev_loss.values))
        
        print("Epoch: %d, global steps: %d, epoch loss: %f"%(epoch, global_steps, epoch_loss))
        if global_steps == args.max_steps:
            break
    plt.ioff()
    return global_steps
        
def predict(args, model, features, labels=None):
    test_x, test_y = Tensor(features), None 
    if labels is not None:
        test_y = Tensor(labels)
    dataloader = DataLoader(test_x, test_y, args.predict_batch_size, shuffle=False)

    print("*********** Predict ***********")
    print("\tExamples num: %d"%(dataloader.data_num))
    print("\tBatch size: %d"%(args.predict_batch_size))

    preds = None
    tot_loss = 0.0 if labels is not None else  -100.0 * dataloader.data_num

    for batch in dataloader():        
        test_preds = model.forward(batch.inputs)
        if labels is not None:
            test_loss = model.loss_layer.loss(test_preds, batch.labels)
            tot_loss += test_loss.values * batch.len
        
        if preds is None:
            preds = test_preds.values
        else:
            preds = np.concatenate((preds, test_preds.values), axis=0)
    
    return preds, tot_loss / dataloader.data_num



def visualize(ax3, features, labels, preds):
    X1 = features[:, :-1]
    X2 = features[:, -1:]

    ax3.scatter(X1, X2, labels)
    ax3.scatter(X1, X2, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str,
                        help="The data dir. Containing train ,test and dev data")
    parser.add_argument("--model_dir", default=None, type=str, 
                        help="The model dir. Set when fine-tuning or predicting.")
    parser.add_argument("--model_save_dir", default="./models/", type=str,
                        help="The dir where model is saved.")
    parser.add_argument("--train", action="store_true", 
                        help="Train or fine-tune a model.")
    parser.add_argument("--predict", action="store_true",
                        help="Use model to predict.")
    parser.add_argument("--draw", action="store_true",
                        help="Draw plot")
    parser.add_argument("--save", action="store_true", 
                        help="Set to store the model.")
    parser.add_argument("--lr", default=3e-5, type=float, 
                        help="learing rate.")
    parser.add_argument("--train_epoch_num", default=3, type=int)
    parser.add_argument("--max_steps", default=0, type=int,
                        help="Max training steps. Overwrite epoch num.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--predict_batch_size", default=32, type=int)
    args = parser.parse_args()

    train_dataset = load_data(os.path.join(args.data_dir, "train.csv"))
    dev_dataset = load_data(os.path.join(args.data_dir, "dev.csv"))
    test_dataset = load_data(os.path.join(args.data_dir, "test.csv"))
    train_features, train_labels = train_dataset[:, :-1], train_dataset[:, -1:]
    test_features, test_labels = test_dataset[:, :-1], test_dataset[:, -1:]
    
    if args.train:
        # draw_scatter(train_features, train_labels)
        if args.model_dir is not None:
            assert "Shouldn't load model when training"
        
        
        net = Network([
                        DenseLayer(200), ReLU(),
                        DenseLayer(100), ReLU(),
                        DenseLayer(100), ReLU(),
                        DenseLayer(80),  ReLU(),
                        DenseLayer(50),  ReLU(),
                        DenseLayer(30),  ReLU(),
                        DenseLayer(1, w_initializer=XavierInitializer())
                        ])
        '''net = Network([
                        DenseLayer(30),  ReLU(),
                        DenseLayer(50),  ReLU(),
                        DenseLayer(20),  ReLU(),
                        DenseLayer(1, w_initializer=XavierInitializer())
                        ])'''
                    
        model = Model(net=net, loss_layer=SquareLoss(), optimizer=SGD(args.lr))
        global_steps = train(args, model, train_features, train_labels, dev_dataset)
        
        print("Global train steps: %d"%(global_steps))
        
        if args.save:
            random.seed(datetime.datetime.now())
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            model_path = os.path.join(args.model_save_dir, "model_%d"%(random.randint(0, 1000)))
            print("Model name: %s"%(model_path.split('/')[-1]))
            model.save(model_path)

    if args.predict:
        draw_scatter(test_dataset[:, :-1], test_dataset[:, -1:])
        if args.model_dir is None and not args.train:
            assert "Set the model dir!"
        if not args.train:
            net = Model.load(args.model_dir)
            model = Model(net=net, loss_layer=SquareLoss(), optimizer=None)
        
        preds, mean_loss = predict(args, model, test_features, test_labels)
        print("Mean loss: %f"%(mean_loss))
    
        if args.draw:
            draw_scatter(test_features, preds, 'g')

if __name__ == "__main__":
    main()