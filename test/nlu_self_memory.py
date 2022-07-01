from pprint import pprint
import json
from paddlenlp import Taskflow

with open('focus.json', 'r', encoding='utf-8') as f:
    schema = json.load(f)
if not schema or "" in schema:
    raise RuntimeError('Drama focus.json not valid, pls refer to above info and try again')

uie = Taskflow('information_extraction', schema=schema)


def nlu_topic(text: [str]) -> list:
    results = uie(text)
    topics = []
    for _result in results:
        topic = {}
        for key, value in _result.items():
            topic[key] = set([entity['text'] for entity in value if entity['probability'] > 0.64])
        topics.append(topic)
    return topics


with open('memory.txt', 'r', encoding='utf-8') as f:
    self_memory_text = [line.strip() for line in f.readlines() if line.strip()]

if len(self_memory_text) == 0:
    print('no data in memory.txt,this is not allowed')

topics = nlu_topic(self_memory_text)
self_memory = []
for i in range(len(self_memory_text)):
    self_memory.append({**{"text": self_memory_text[i]}, **topics[i]})

pprint(self_memory)
