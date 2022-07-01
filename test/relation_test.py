from pprint import pprint

with open('relations.txt', 'r', encoding='utf-8') as f:
    relations_text = [line.strip() for line in f.readlines() if line.strip()]

if len(relations_text) == 0 or len(relations_text) % 2 != 0:
    print('no data in relations.txt or the number of lines is not even')

relations = dict(zip(relations_text[::2], relations_text[1::2]))

pprint(relations)
