```bash
pip install -r requirements.txt
jupyter notebook
```

Plocka word vectors:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
```


Skapa wiki corpus:
```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python make_wiki_corpus.py enwiki-latest-pages-articles.xml.bz2 wiki_en.txt
```