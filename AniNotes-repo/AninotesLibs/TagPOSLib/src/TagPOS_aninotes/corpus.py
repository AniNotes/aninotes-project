'''import os, os.path
import nltk.data

path = os.path.expanduser('~/nltk_data')
print(path in nltk.data.path)'''

import requests
from bs4 import BeautifulSoup

def get_transcript(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    tscpt = None
    div_tags = soup.find_all('div')
    for tag in div_tags:
        tag_str = tag.string
        if not tag_str:
            continue
        if tag_str.count("\n") < len(tag_str).bit_length():
            continue
        tscpt = tag_str
        break
    tscpt_lines = tscpt.split("\n")
    for line in tscpt_lines:
        print(line + "\n")

def main():
    '''url = "https://www.khanacademy.org/math/linear-algebra/alternate-bases/othogonal-complements/v/linear-algebra-orthogonal-complements"
    tscpt_lines = get_transcript(url)
    for line in tscpt_lines:
        print(line + "\n")'''
    
    url = "https://www.khanacademy.org/math"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup)
    tags = soup.find_all('span')
    for tag in tags:
        #print(tag)
        tag_string = tag.string
        if tag_string:
            print(tag)
            print(tag_string)
            print("\n\n")


if __name__ == "__main__":
    main()