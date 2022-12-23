'''
    {
        "url": "https://cs.stanford.edu/people/rak248/VG_100K/2356347.jpg", 
        "image_id": 2356347, 
        "paragraph": "A large building with bars on the windows in front of it. 
        There is people walking in front of the building. There is a street in front of the building with many cars on it."
    }
    
    to

    {
        "image_path": "/path/to/image", 
        "id": 2356347, 
        "paragraph": "A large building with bars on the windows in front of it. 
        There is people walking in front of the building. There is a street in front of the building with many cars on it."å
    }

    to load & set data

'''
import json
import pprint
import requests
import re
from pathlib import Path
from config import parse_args

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)

def preprocess_json(parags, train_ids, valid_ids, test_ids):
    '''
        download img to ./data/processed/img with {id}.jpg
        make paragraphs three json files(train/valid/test) with image_path

        {
            "url": "https://cs.stanford.edu/people/rak248/VG_100K/2356347.jpg", 
            "image_id": 2356347, 
            "paragraph": "A large building with bars on the windows in front of it. 
            There is people walking in front of the building. There is a street in front of the building with many cars on it."
        }
    '''
    train_file, valid_file, test_file = [], [], []
    for i, parag in enumerate(parags):

        url = parag['url']
        id = parag['image_id']
        paragraph = parag['paragraph']
        image_path = f'./data/processed/img/{id}.jpg'

        ## download img to path with URL
        # response = requests.get(url)
        # with open(image_path, "wb") as f:
        #     f.write(response.content)

        ## processed json files
        new_parag = {
            "url": url,
            "id": id,
            "paragraph": paragraph,
            "image_path": image_path
        }

        if id in train_ids:
            train_file.append(new_parag)
        elif id in valid_ids:
            valid_file.append(new_parag)
        else:
            test_file.append(new_parag)

    return train_file, valid_file, test_file

if __name__ == '__main__':
    args = parse_args()

    parag_path = './data/raw/all_paragraphs.json'
    all_parag = load_json_file(parag_path)

    print(f' There are {len(all_parag)} data')

    ## splits
    train_ids = load_json_file(args.raw_train_split) 
    valid_ids = load_json_file(args.raw_valid_split) 
    test_ids = load_json_file(args.raw_test_split) 

    train_file, valid_file, test_file = preprocess_json(
        parags = all_parag,
        train_ids = train_ids,
        valid_ids = valid_ids,
        test_ids = test_ids,
    )        

    save_json(args.train_file, train_file)
    save_json(args.valid_file, valid_file)
    save_json(args.test_file, test_file)
    print('processed files saved!')









