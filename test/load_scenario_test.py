import xlrd
import json
from pprint import pprint

data = xlrd.open_workbook('scenarios.xlsx')
rules = {}
last_turn_memory_template = {}
for name in data.sheet_names():
    table = data.sheet_by_name(name)
    nrows = table.nrows
    cols = table.ncols

    rules[name] = {}
    last_turn_memory_template[name] = {}
    for k in range(1, cols):
        last_turn_memory_template[name][table.cell_value(0, k)] = {"text": [], "talker": ['你']}
        rules[name][table.cell_value(0, k)] = {}
        for i in range(1, nrows):
            if table.cell_value(i, k):
                rules[name][table.cell_value(0, k)][table.cell_value(i, 0)] = table.cell_value(i, k)

with open('last_turn_memory_template.json', 'w', encoding='utf-8') as f:
    json.dump(last_turn_memory_template, f, ensure_ascii=False)

pprint(rules)

print("test '海达':")
actions = []
actions.extend(rules['welcome']['陌生人'].get('海达', '').split('\n'))
if not actions:
    actions = ['']
print(actions)

print("test '侯赛因':")
actions = []
actions.extend(rules['welcome']['陌生人'].get('侯赛因', '').split('\n'))
if not actions:
    actions = ['']
print(actions)