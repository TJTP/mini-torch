import numpy as np 
import argparse
import os 
import random 
import datetime

from numpy.core.defchararray import mod

from mini_torch.model import Model
from mini_torch.tensor import Tensor
from mini_torch.network import Network
from mini_torch.layer import DenseLayer, ReLU
from mini_torch.loss import SquareLoss
from mini_torch.optimizer import SGD
from mini_torch.data_loader import DataLoader
from mini_torch.initializer import HeInitializer, XavierInitializer

def train(args, model, dataset):
    train_x, train_y = Tensor(dataset[:, :-1]), Tensor(dataset[:, -1:])
    iterator = DataLoader(args.train_batch_size)
    if args.max_steps > 0:
        args.train_epoch_num = args.max_steps // args.train_batch_size + 1
    
    print("*********** Train ***********")
    print("\tExamples num: %d"%(len(dataset)))
    print("\tEpoch num: %d"%(args.train_epoch_num))
    print("\tBatch size: %d"%(args.train_batch_size))
    print("\tBatch num: %d"%(len(train_y) // args.train_batch_size + 1 if len(train_y) % args.train_batch_size != 0 else len(train_y) // args.train_batch_size))

    global_steps = 0
    for epoch in range(args.train_epoch_num):
        for batch in iterator(train_x, train_y):
            model.zero_grad()
            preds = model.forward(batch.inputs)
            loss = model.loss_layer.loss(preds, batch.labels)
            #print(loss.values)
            loss.backward()
            model.step()
            global_steps += 1
            if global_steps == args.max_steps:
                break
        if global_steps == args.max_steps:
            break
    
    return global_steps
        
    

def predict(args, model, test_dataset):
    print("*********** Predict ***********")
    print("\tExamples num: %d"%(len(test_dataset)))
    print("\tBatch size: %d"%(args.predict_batch_size))

    test_x, test_y = Tensor(test_dataset[:, :-1]), Tensor(test_dataset[:, -1:])
    model.set_status(is_training=False)
    iterator = DataLoader(args.train_batch_size)
    preds = None
    tot_loss = 0.0
    for batch in iterator(test_x, test_y):        
        test_preds = model.forward(batch.inputs)
        test_loss = model.loss_layer.loss(test_preds, batch.labels)
        tot_loss += test_loss.values
        print("Current batch loss: %f"%(test_loss.values))
        if preds is None:
            preds = test_preds.values
        else:
            preds = np.concatenate((preds, test_preds.values), axis=0)
    model.set_status(is_training=True)
    return preds, tot_loss



def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=",")
    return data

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
    parser.add_argument("--lr", default=3e-5, type=float, 
                        help="learing rate.")
    parser.add_argument("--train_epoch_num", default=3, type=int)
    parser.add_argument("--max_steps", default=0, type=int,
                        help="Max training steps. Overwrite epoch num.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--predict_batch_size", default=32, type=int)
    args = parser.parse_args()

    
    model_path = None
    if args.train:
        if args.model_dir is not None:
            #fine-tune the model
            fine_tune_dataset = load_data(args.data_dir + "fine_tune.csv")
            net = Model.load(args.model_dir)
            model = Model(net=net, loss_layer=SquareLoss(), optimizer=SGD(args.lr))
            global_steps = train(args, model, fine_tune_dataset)
            
        else:
            train_dataset = load_data(args.data_dir + "train.csv")
            net = Network([DenseLayer(3, w_initializer=HeInitializer()), 
                           ReLU(), 
                           DenseLayer(4, w_initializer=HeInitializer()), 
                           ReLU(), 
                           DenseLayer(1, w_initializer=HeInitializer())])

            model = Model(net=net, loss_layer=SquareLoss(), optimizer=SGD(args.lr))
            global_steps = train(args, model, train_dataset)
        
        print("Global train steps: %d"%(global_steps))
        random.seed(datetime.datetime.now())
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        model_path = os.path.join(args.model_save_dir, "model_{}".format(random.randint(0, 200)))
        print(model_path)
        model.save(model_path)

    if args.predict:
        if args.model_dir is None and not args.train:
            assert "Set the model dir!"
        if args.train:
            net = Model.load(model_path)
        else:
            net = Model.load(args.model_dir)
        model = Model(net=net, loss_layer=SquareLoss(), optimizer=None)
        test_dataset = load_data(args.data_dir + "test.csv")
        preds, tot_loss = predict(args, model, test_dataset)
        print("Total loss: %f"%(tot_loss))


if __name__ == "__main__":
    main()