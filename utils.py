import re
import numpy as np


def load_articles(filename):
    articles = []
    with open(filename, 'r') as f:
        article = ''
        for line in f:
            if re.match('^ = [\w ]* = $', line) and len(article) > 2:
                articles.append(article)
                article = ''

            article += line

    return articles


def load_goblet(filename):
    chapters = []
    with open(filename, 'r') as f:
        chapter = ''
        for line in f:
            if re.match('^CHAPTER', line) and len(chapter) > 2:
                chapters.append(chapter)
                chapter = ''

            chapter += line.strip()

    return chapters


def get_encoding(articles):
    chars = list(set("".join(articles)))

    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for i, ch in enumerate(chars)}

    return char2int, int2char
