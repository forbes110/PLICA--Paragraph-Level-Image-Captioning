## ref website: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html

## env
pip install -r requirements.txt

## load raw data 
if [ ! -f all_paragraphs.zip ]; then
    wget https://www.dropbox.com/s/ue1pru8fpkehrxg/all_paragraphs.zip?dl=1 -O all_paragraphs.zip
    unzip all_paragraphs.zip -d ./data/raw
fi

python3 set_data.py
