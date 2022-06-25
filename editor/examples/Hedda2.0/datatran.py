import re

with open('intents.yml', 'r', encoding='utf-8') as f:
    texts = [line for line in f.readlines() if line.strip()]

with open('intents_new.yml', 'w', encoding='utf-8') as f:
    for text in texts:
        _text = re.search(f'[\u4e00-\u9fa5].+', text)
        if _text:
            _text = _text.group()
            print(_text)
            text = text[:text.index(_text[0])]+'陌生人说：“'+_text+'”'
            print(text)
        f.write(text + "\n")
