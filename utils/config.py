import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(description = "Lazy LLM for SMO")

    parser.add_argument('-d','--dataset', type = str, default = "", help = "choose the dataset")

    args = parser.parse_args()

    return args

parse_arguments()


