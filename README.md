# 低质词

- 概念：汉语中不合理的, 不适合出现的（从用户体验的角度），含有错别字（用户把输入法当作权威）/不成词的词语组合(如周嗯来->周恩来， 财大器粗->财大气粗，巧克-> 巧克力，流奶->牛奶，流浪织女->牛郎织女)(同音错别字，不成词错别字，方言混淆音，奇怪的ngram词组)

- 来源：语言模型和词库的训练语料大部分来自于用户输入

  - 绝大多数用户的聊天场景是非正式的（**输入习惯比较随意**，有时候会根据自己的错误认知和习惯输入，因此语料不像新闻那样干净）；
  - 采集用户日志信息时没有考虑到用户的一些行为；
  - 且采集语料后分词是上屏 自然分词，多了很多不成词

  从而导致会产生很多低质量的语料，而前期数据清洗不完全，使得很多脏数据也用于训练。

- 风险/解决价值：在词库和语言模型中过多的低质词将降低输入法出词和联想质量，从而影响用户体验（低质量得出词会让人觉得输入法不专业，例子：有部分人将输入法当作词典，查阅拼音，成语，诗词）

## 读取文本文件
ngram-arpa 文件读取

class ngram_data:
    def __init__(self, logp, chinese, pinyin, backoff,ngram):
        self.logp = logp
        self.chinese = chinese
        self.pinyin = pinyin
        self.backoff = backoff
        self.ngram = ngram
    
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
    test = len(line.split('\t'))
    if test < 2:
        continue
    elif test == 2:
        logp, mid = line.split('\t')
    elif test == 3:
        logp, mid, backoff = line.split('\t')
    chinese, pinyin = mid.split('\\1')
    pinyin = ''.join(pinyin.split(' '))
    ngram = ngram_data(logp, chinese, pinyin, backoff,ngram)
    ls_ngram.append(ngram)
    
    
        
```
## 同音-编辑距离的低质词探查
本项目将探查拼音相同，编辑距离接近的词，这么做的原因是当时实习的项目是拼音输入法，训练语言模型的语料库来源于用户输入，常常有用户键入正确的拼音而点选错误的词，因此语料库含有大量这样的错别词。
```python
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


```

如图是用同拼音-编辑距离方法找到的低质词，该方法效率还是挺高的，时间复杂度在O(n)，当时实习的语言模型是千万级别的数据不到一小时就跑完了


