import requests
from bs4 import BeautifulSoup
import nltk

def get_pos_of_words(words):
    n_pos_words = {word : [] for word in words}
    stored_pos = open("pos.txt").read().splitlines() # Lines will look like: "word pos1 pos2 pos3"
    file_words = set()
    for pos in stored_pos:
        pos.strip()
        pos_info = pos.split(" ")
        word = pos_info[0]
        file_words.add(word)
        pos_list = pos_info[1:]
        n_pos_words[word] = pos_list
    pos_set = {
    "noun",
    "adjective",
    "adverb",
    "conjunction",
    "preposition",
    "pronoun",
    "verb",
    "definite_article",
    "indefinite_article",
    "determiner",
    "indefinite marker",
    "modal verb",
    "auxiliary verb"
    }
    with open("pos.txt", "a") as f:
        for word in words:
            if word not in file_words:
                cache = "http://webcache.googleusercontent.com/search?q=cache:"
                url = cache + f"https://www.oxfordlearnersdictionaries.com/us/definition/english/{word}?q={word}"
                #url = cache + "https://www.oxfordlearnersdictionaries.com/us/definition/english/this_2"
                page = requests.get(url)
                if page.status_code != 200:
                    url = cache + f"https://www.oxfordlearnersdictionaries.com/us/definition/english/{word}_1?q={word}"
                    page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                #print(soup)
                title_tag = soup.find('title')
                title_str = title_tag.string
                title = title_str[ : title_str.index("-") - 1].split(" ")
                if title[0] == word:
                    n_pos_words[word].append("_".join(title[1 : ]))
                    a_tags = soup.find_all('a')
                    for tag in a_tags:
                        href = tag.get('href')
                        if href:
                            if href.startswith(f"https://www.oxfordlearnersdictionaries.com/definition/english/{word}"):
                                try:
                                    pos_int = int(href[href.index(f"{word}_") + len(word) + 1 : ])
                                    if pos_int != 1:
                                        tag_title = tag.get('title').split(" ")
                                        pos = "_".join(tag_title[1 : tag_title.index("definition")])
                                        if pos in pos_set:
                                            n_pos_words[word].append(pos)
                                except:
                                    continue
                    spc = " "
                    pos_str = f"{word} {spc.join(n_pos_words[word])}\n"
                    #f.write(pos_str)
                else:
                    print("good")
                    print(soup)
        f.close()
    return n_pos_words

def category_split(words, dct, ctgs, just_inds = False):
    inds = [i for i in range(len(words)) if words[i] in dct and dct[words[i]] in ctgs] + [len(words)]
    if just_inds:
        return inds
    split_words = [words[inds[i - 1] + 1 : inds[i]] for i in range(1, len(inds))]
    return inds, split_words

def get_words(text):
    punc_str = ".,?!:;"
    txt_chars = list(text)
    for i, ch in enumerate(txt_chars):
        if ch in punc_str:
            text_chars.insert(i + 1, " ")
    spaced_txt = "".join(txt_chars)
    txt_itms = spaced_txt.split(" ")
    words, punc, word_inds, punc_inds = []
    for i, itm in enumerate(txt_itms):
        if itm in punc_str:
            punc.append(itm)
            punc_inds.append(i)
        else:
            words.append(itm)
            word_inds.append(i)
    return words, punc, word_inds, punc_inds

def tag_for_pos(text):
    words = get_words(text)
    pos_of_words = {word : "" for word in words}
    n_pos_words = get_pos_of_words(words)
    for word in words:
        if len(n_pos_words[word]) == 1:
            pos_of_words[word] = n_pos_words[word][0]

def all_capitals(word):
    cap = "AQWERTYUIOPASDFGHJKLZXCVBNM"
    for ch in word:
        if ch not in cap:
            return False
    return True

def sentence_split(text):
    sentence_enders = {'.' : "sentence_end", '!' : "sentence_end", '?' : "sentence_end"}
    ender_inds = category_split(text, sentence_enders, "sentence_end", just_inds = True)
    sentences = []
    sentence = []
    for i in range(len(text)):
        token = text[i]
        sentence.append(token)
        if i in ender_inds:
            prev_token = text[i - 1]
            if not(prev_token in ('Mr', 'Ms', 'Mrs', 'Dr', 'Mx') or (len(prev_token) == 1 and all_capitals(prev_token))):
                sentences.append(sentence)
                sentence = []
    return sentences

import time

if __name__ == "__main__":
    #print(get_pos_of_words(["claim"]))
    #print(category_split("the feisty tiger ate a delicious baby animal in between a small grove and an enormously high rock.".split(" "), {"the" : "article", "a" : "article", "an" : "article"}, {"article"}))
    #print(list(filter(None, "".join([" ", "-"][b] for b in [False, True, True, False, False, False, True, False]).split(" "))))
    '''l = [i for i in range(100000000)]
    t1 = time.perf_counter()
    l1 = l[::-1]
    print(time.perf_counter() - t1)
    t2 = time.perf_counter()
    l2 = reversed(l)
    print(time.perf_counter() - t2)
    t3 = time.perf_counter()
    l.reverse()
    print(time.perf_counter() - t3)'''

    '''txt = nltk.corpus.gutenberg.words('austen-emma.txt')
    txt = list(txt[11:])
    _, sentences = category_split(txt, {'.' : "sentence_end", '!' : "sentence_end", '?' : '.' : "sentence_end"}, "sentence_end")
    print(txt[2390:2420])
    fixed_sentences = []
    i = 0
    while i in range(len(sentences)):
        sentence = sentences[i]
        words = sentence.split(' ')
        if (final_word := words[-1]) in ('Mr', 'Ms', 'Mrs', 'Dr', 'Mx') or (word != 'I' and all_capitals(final_word)):
            fixed_sentences = sentence + "."
    sentence_lengths = [len(sentence) for sentence in sentences]
    sentence_length_counts = [0 for _ in range(max(sentence_lengths))]
    for i, length in enumerate(sentence_lengths):
        #if length == 1:
        #    print(sentences[i])
        if length != 0:
            sentence_length_counts[length - 1] = sentence_length_counts[length - 1] + 1
    print(sentence_length_counts)'''

    '''text = "I am writing random gibberish , and I am not sure what is going to happen . Here we go ; writing another sentence , eh ! How are you ? Oh my , and now we continue . What about a test from Mr. Johnson . He abbreviates to Mr . J . every so often . I think we're done now!".split(' ')
    sentences = sentence_split(text)
    for sentence in sentences:
        print(sentence)'''

    text = nltk.corpus.gutenberg.words('austen-emma.txt')
    text = list(text[11:])
    sentences = sentence_split(text)
    sentence_lengths = [len(sentence) for sentence in sentences]
    sentence_length_counts = [0 for _ in range(max(sentence_lengths))]
    for i, length in enumerate(sentence_lengths):
        if length == 1:
            print(sentences[i])
        if length != 0:
            sentence_length_counts[length - 1] = sentence_length_counts[length - 1] + 1
    print(sentence_length_counts)
