#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict


class ngram_data:
    def __init__(self, logp, chinese, pinyin, backoff,ngram):
        self.logp = logp
        self.chinese = chinese
        self.pinyin = pinyin
        self.backoff = backoff
        self.ngram = ngram


def edit_distance(w1, w2):
    '''
    Calculate how many char in different between two words.
    '''

    distance = 0
    for tuple_ in list(zip(w1, w2)):
        if len(set(tuple_)) > 1:
            distance += 1
    return distance


ngram_model_add = r'/Users/yun/Desktop/Project/low-quality-word/lm_ngram.arpa'
ngram_model = open(ngram_model_add, 'r', encoding='utf-8').readlines()
ls_ngram = []
for line in ngram_model:
    line = line.strip()
    if not line:
        continue
    if line == '\\1-grams:':
        ngram = 1
        continue
    if line == '\\2-grams:':
        ngram = 2
        continue
    if line == '\\3-grams:':
        ngram = 3
        continue
    test = len(line.split('\\t'))
    if test < 2:
        continue
    elif test == 2:
        logp, mid = line.split('\\t')
    elif test == 3:
        logp, mid, backoff = line.split('\\t')
    chinese, pinyin = mid.split('\\1')
    pinyin = ''.join(pinyin.split(' '))
    chinese = ''.join(chinese.split(' '))
    ngram_word = ngram_data(logp, chinese, pinyin, backoff, ngram)
    ls_ngram.append(ngram_word)

uni_word = []
n_word = []
for word in ls_ngram:
    if word.ngram == 1:
        uni_word.append(word)
    else:
        n_word.append(word)

d_pinyin_word = defaultdict(list)
d_unigram_ngram = defaultdict(list)
unigram_pinyin = [unigram.pinyin for unigram in uni_word]


for unigram in uni_word:
    d_pinyin_word[unigram.pinyin].append(unigram)
for ngram in n_word:
    if ngram.pinyin in d_pinyin_word:
        for unigram in d_pinyin_word[ngram.pinyin]:
            if 1 <= edit_distance(unigram.chinese, ngram.chinese) <= 2:
                d_unigram_ngram[unigram].append(ngram)

for key in d_unigram_ngram:
    print(key.chinese)
    for item in d_unigram_ngram[key]:
        print('\t', item.chinese, item.ngram)

