def nlu_topic(test):
    ners, topics = [], []
    for _result in test:
        ner, topic = [], []
        for key, value in _result.items():
            _ner = [entity['text'] for entity in value if entity['probability'] > 0.64]
            if _ner:
                ner.extend(_ner)
                topic.append(key)
        ners.append(ner)
        topics.append(topic)
    return ners, topics

test1 = [{'作品': [{'text': '海达·高布乐', 'start': 8, 'end': 14, 'probability': 0.6955123581948648}]}]
test2 = [{}]
test3 = [{'作品': [{'text': '玩偶之家', 'start': 9, 'end': 13, 'probability': 0.9072484369487555, 'relations': {'作者': [{'text': '张爱玲', 'start': 4, 'end': 7, 'probability': 0.9972492544409306}], '情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.8659434160334136}]}}], '作者': [{'text': '张爱玲', 'start': 4, 'end': 7, 'probability': 0.9981177894717916, 'relations': {'作品': [{'text': '玩偶之家', 'start': 9, 'end': 13, 'probability': 0.9581599080458858}]}}]}]

print(nlu_topic(test1))
print(nlu_topic(test2))
print(nlu_topic(test3))