## ref website: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html

## env
pip install -r requirements.txt
brew install java

## load raw data 
if [ ! -f all_paragraphs.zip ]; then
    wget https://www.dropbox.com/s/ue1pru8fpkehrxg/all_paragraphs.zip?dl=1 -O all_paragraphs.zip
    unzip all_paragraphs.zip -d ./data/raw
fi

## download processed image data
if [ ! -f visual_genome.zip ]; then
    wget https://www.dropbox.com/s/z7os20crg40aupb/visual_genome.zip?dl=1 -O visual_genome.zip
    unzip -j visual_genome.zip -d ./data/processed/img
fi

## process paragraph data
python3 set_data.py


