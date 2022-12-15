'''
    for params setting
'''

from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser()

    ## processed
    parser.add_argument(
        "--train_file", default='./data/processed/paragraphs/train_file.json', type=str
    )
    parser.add_argument(
        "--valid_file", default='./data/processed/paragraphs/valid_file.json', type=str
    )
    parser.add_argument(
        "--test_file", default='./data/processed/paragraphs/test_file.json', type=str
    )

    ## models
    parser.add_argument(
        "--batch_size", default=4, type=int
    )



    args = parser.parse_args()
    return args

