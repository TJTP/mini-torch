import numpy as np 
import argparse
import os 
import random 
import datetime
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

from mini_torch.tensor import Tensor
from mini_torch.layer import DenseLayer
from mini_torch.initializer import XavierInitializer
from mini_torch.activation import ReLU
from mini_torch.network import Network
from mini_torch.model import Model
from mini_torch.loss import SquareLoss
from mini_torch.optimizer import SGD
from mini_torch.data_loader import DataLoader

from utils import draw_scatter, draw_2d

def train(args, model, dataset, test_dataset=None):
    train_x_numpy, train_y_numpy = dataset[:, :-1], dataset[:, -1:]
    train_x, train_y = Tensor(train_x_numpy), Tensor(train_y_numpy)

    if test_dataset is not None:
        test_x_numpy, test_y_numpy = test_dataset[:, :-1], test_dataset[:, -1:]
        test_x, test_y = Tensor(test_x_numpy), Tensor(test_y_numpy)
    
    if args.max_steps > 0:
        args.train_epoch_num = args.max_steps // args.train_batch_size + 1

    dataloader = DataLoader(train_x, train_y, args.train_batch_size)
    
    print("*********** Train ***********")
    print("\tExamples num: %d"%(len(dataset)))
    print("\tEpoch num: %d"%(args.train_epoch_num))
    print("\tBatch size: %d"%(args.train_batch_size))
    print("\tBatch num: %d"%(dataloader.len))
    
    fig = plt.figure()
    ax3 = Axes3D(fig)
    plt.ion()
    plt.show()

    global_steps = 0
    for epoch in range(args.train_epoch_num):
        for batch in dataloader():
            model.zero_grad()
            preds = model.forward(batch.inputs)
            loss = model.loss_layer.loss(preds, batch.labels)
            loss.backward()
            model.step()
            global_steps += 1
            if global_steps % dataloader.len == 0:
                print("Epoch: %d, global steps: %d, current batch loss: %f"%(epoch, global_steps, loss.values))
            if global_steps == args.max_steps:
                break
        if test_dataset is not None and global_steps % 5 == 0:
            test_preds = model.forward(test_x)

            plt.cla()
            visualize(ax3, test_x_numpy, test_y_numpy, test_preds.values)
            plt.pause(0.1)

            test_loss = model.loss_layer.loss(test_preds, test_y)
            print("****Draw on epoch-%d, test loss: %f****"%(epoch, test_loss.values))
        if global_steps == args.max_steps:
            break
    plt.ioff()
    return global_steps
        
def predict(args, model, test_dataset):
    print("*********** Predict ***********")
    print("\tExamples num: %d"%(len(test_dataset)))
    print("\tBatch size: %d"%(args.predict_batch_size))

    test_x, test_y = Tensor(test_dataset[:, :-1]), Tensor(test_dataset[:, -1:])
    dataloader = DataLoader(test_x, test_y, args.predict_batch_size, shuffle=False)
    preds = None
    tot_loss = 0.0

    for batch in dataloader():        
        test_preds = model.forward(batch.inputs)
        test_loss = model.loss_layer.loss(test_preds, batch.labels)
        tot_loss += test_loss.values * batch.len
        
        if preds is None:
            preds = test_preds.values
        else:
            preds = np.concatenate((preds, test_preds.values), axis=0)
    return preds, tot_loss / dataloader.data_num

def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=",")
    return data

def visualize(ax3, features, labels, preds):
    X1 = features[:, :-1]
    X2 = features[:, -1:]

    ax3.scatter(X1, X2, labels)
    ax3.scatter(X1, X2, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str,
                        help="The data dir. Containing train ,test and fine-tune data")
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
    test_dataset = load_data(os.path.join(args.data_dir, "test.csv"))
    
    draw_scatter(train_dataset[:, :-1], train_dataset[:,-1:])
    if args.train:
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
        global_steps = train(args, model, train_dataset, test_dataset)
        
        print("Global train steps: %d"%(global_steps))
        
        if args.save:
            random.seed(datetime.datetime.now())
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            model_path = os.path.join(args.model_save_dir, "model_%d"%(random.randint(0, 1000)))
            print("Model name: %s"%(model_path.split('/')[-1]))
            model.save(model_path)

    if args.predict:
        if args.model_dir is None and not args.train:
            assert "Set the model dir!"
        if not args.train:
            net = Model.load(args.model_dir)
            model = Model(net=net, loss_layer=SquareLoss(), optimizer=None)
        
        preds, mean_loss = predict(args, model, train_dataset)
        print("Mean loss: %f"%(mean_loss))
    
        if args.draw:
            draw_scatter(train_dataset[:, :-1], preds, 'g')

if __name__ == "__main__":
    main()