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
    parser.add_argument("--fine_tune", action="store_true",
                        help="Generate data for fine-tuning")
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
    #print(examples)
    random.shuffle(examples)
    #print(examples)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.fine_tune:
        np.savetxt(args.output_dir + "fine_tune.csv", examples, fmt="%f", delimiter=',')
    else:
        num_train = int(args.ratio * len(examples))
        train_dataset = examples[:num_train]
        test_dataset = examples[num_train:]
        np.savetxt(args.output_dir + "train.csv", train_dataset, fmt="%f", delimiter=',')
        np.savetxt(args.output_dir +"test.csv", test_dataset, fmt="%f", delimiter=',')
    

if __name__ == "__main__":
    main()
