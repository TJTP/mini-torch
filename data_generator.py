import numpy as np 
from math import sin, cos 
import random
import argparse
import os 

def f(x_1, x_2):
    return sin(x_1) - cos(x_2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", default=200, type=int, 
                        help="Num of examples")
    parser.add_argument("--output_dir", default="./data/", type=str, 
                        help="Output dir.")
    parser.add_argument("--ratio", default=0.8, type=float,
                        help="Division ratio of train and test data")
    parser.add_argument("--extra_ratio", default=0, type=float,
                        help="Data ratio on special interval [-2, 2]. Value should in [0, 0.5]")
    args = parser.parse_args()
    
    examples = []
    for _ in range(args.num):
        example = []
        x_1 = random.uniform(-5, 5)
        x_2 = random.uniform(-5, 5)
        example.append(x_1)
        example.append(x_2)
        example.append(f(x_1, x_2))
        examples.append(example)
    random.shuffle(examples)
    num_train = int(args.ratio * len(examples))
    train_dataset = examples[:num_train]
    test_dataset = examples[num_train:]
    assert 0 <= args.extra_ratio <= 0.5
    for _ in range(int(args.extra_ratio * num_train)):
        example = []
        x_1 = random.uniform(-2, 2)
        x_2 = random.uniform(-2, 2)
        example.append(x_1)
        example.append(x_2)
        example.append(f(x_1, x_2))
        train_dataset.append(example)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    np.savetxt(args.output_dir + "train.csv", train_dataset, fmt="%f", delimiter=',')
    np.savetxt(args.output_dir +"test.csv", test_dataset, fmt="%f", delimiter=',')
    

if __name__ == "__main__":
    main()
