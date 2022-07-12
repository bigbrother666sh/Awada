import os
import json
#import time
#import re
#import wechaty
#from paddlenlp import Taskflow
#from typing import Optional
import xlrd
"""
from wechaty import (
    Contact,
    MessageType,
    WechatyPlugin,
    Message,
    WechatyPluginOptions
)
from wechaty_puppet import get_logger
"""
#from utils.DFAFilter import DFAFilter
#from utils.rasaintent import RasaIntent
from plugins.inspurai.inspurai import Yuan


class Drama:
    def __init__(
            self,
            configs: str = 'drama_configs',
    ) -> None:

        # 1. create the cache_dir
        self.config_url = configs
        self.config_files = os.listdir(self.config_url)
        """
        if len(self.config_files) < 4:
            raise RuntimeError('Drada plugin config_files not enough, pls add and try again')
        """

        # 4. load scenario rule-table
        self.scenarios = self._load_scenarios()
        if self.scenarios is None:
            raise RuntimeError('Drada scenarios.xlsx not valid, pls refer to above info and try again. make sure at lease one scenario is well defined.')

        # 5. load self-memory systems: self_memory\relations\user_memory\last_turn_memory
        """
        with open(os.path.join(self.config_url, 'focus.json'), 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        if not self.schema or "" in self.schema:
            raise RuntimeError('Drama focus.json not valid, pls refer to above info and try again')

        self.uie = Taskflow('information_extraction', schema=self.schema, task_path='uie/checkpoint/model_best')

        self.self_memory = self._load_memory()
        if not self.self_memory:
            raise RuntimeError('Drada memory.txt not valid, pls refer to above info and try again')
        """
        if "relations.txt" in self.config_files:
            self.relations = self._load_relations()
        else:
            self.relations = {}

        if "memory.json" in self.config_files:
            with open(os.path.join(self.config_url, 'memory.json'), 'r', encoding='utf-8') as f:
                self.memory = json.load(f)
        else:
            self.memory = dict.fromkeys(self.scenarios.keys(), [])

        # 6. initialize yuan-api
        self.yuan = Yuan(engine='dialog',
                         temperature=1,
                         max_tokens=150,
                         input_prefix='',
                         input_suffix='',
                         output_prefix='',
                         output_suffix='',
                         append_output_prefix_to_query=False,
                         topK=3,
                         topP=0.9,
                         frequencyPenalty=1.2,)

    def _file_check(self) -> None:
        """check the config file"""

        if "focus.json" not in self.config_files:
            raise RuntimeError(f'config file url:/{self.config_url} does not have focus.json!')

        if "memory.txt" not in self.config_files:
            raise RuntimeError(f'config file url:/{self.config_url} does not have memory.txt!')

        if "scenarios.xlsx" not in self.config_files:
            raise RuntimeError(f'config file url:/{self.config_url} does not have scenarios.xlsx!')

        if "relations.txt" not in self.config_files:
            raise RuntimeWarning(f"config file url:/{self.config_url} does not have relations.txt. however you can go without it, but it's not recommended!")

    def nlu_topic(self, text: [str]) -> list:
        results = self.uie(text)
        topics = []
        for _result in results:
            topic = {}
            for key, value in _result.items():
                topic[key] = set([entity['text'] for entity in value if entity['probability'] > 0.58])
            topics.append(topic)
        return topics

    def _load_memory(self) -> list:
        """load the memory data"""
        memory_file = os.path.join(self.config_url, 'memory.txt')

        with open(memory_file, 'r', encoding='utf-8') as f:
            self_memory_text = [line.strip() for line in f.readlines() if line.strip()]

        if len(self_memory_text) == 0:
            raise RuntimeError('no data in memory.txt,this is not allowed')

        topics = self.nlu_topic(self_memory_text)
        self_memory = []
        for i in range(len(self_memory_text)):
            self_memory.append({**{"text": self_memory_text[i]}, **topics[i]})

        return self_memory

    def _load_relations(self) -> dict:
        """load the relations data"""
        relations_file = os.path.join(self.config_url, 'relations.txt')

        with open(relations_file, 'r', encoding='utf-8') as f:
            relations_text = [line.strip() for line in f.readlines() if line.strip()]

        if len(relations_text) == 0 or len(relations_text) % 2 != 0:
            raise RuntimeError('no data in relations.txt or the number of lines is not even')

        return dict(zip(relations_text[::2], relations_text[1::2]))

    def _load_scenarios(self) -> dict:
        """load the scenarios data"""
        scenarios_file = os.path.join(self.config_url, 'scenarios.xlsx')
        data = xlrd.open_workbook(scenarios_file)

        rules = {}
        for name in data.sheet_names():
            table = data.sheet_by_name(name)
            nrows = table.nrows
            cols = table.ncols

            rules[name] = {}
            for k in range(1, cols):
                rules[name][table.cell_value(0, k)] = {}
                for i in range(1, nrows):
                    if table.cell_value(i, k):
                        rules[name][table.cell_value(0, k)][table.cell_value(i, 0)] = table.cell_value(i, k)
        return rules

    def soul(self, scenario: str, character: str) -> str:
        memory_text = ''
        for i in range(len(self.memory[scenario])-1, -1, -1):
            memory_text = self.memory[scenario][i] + memory_text
            if len(memory_text) >= 150:
                break

        prompt = self.relations.get("你", "") + self.relations.get(character, '') + self.scenarios[scenario][character].get('DESCRIPTIONTEXT', '') + memory_text + "你说：“"

        for i in range(7):
            reply = self.yuan.submit_API(prompt, trun="”")
            if not reply or reply == "somethingwentwrongwithyuanservice" or reply == "请求异常，请重试":
                continue
            if len(reply) <= 5 or reply not in memory_text:
                break

        if not reply or reply == "somethingwentwrongwithyuanservice" or reply == "请求异常，请重试":
            return reply

        self.memory[scenario].append(f"你说：“{reply}”")
        return reply


if __name__ == '__main__':
    caixiao = Drama('caixiao')
    sunruo = Drama('sunruo')
    #zhangjiayi = Drama('drama_configs/zhangjiayi')

    #gfw = DFAFilter()
    #gfw.parse()
    #rasa = RasaIntent()

    while True:
        text = sunruo.soul('awake', 'caixiao')
        print(f"孙若说：“{text}”")
        caixiao.memory['awake'].append(f"孙若说：“{text}”")
        text = caixiao.soul('awake', 'sunruo')
        print(f"蔡晓说：“{text}”")
        sunruo.memory['awake'].append(f"蔡晓说：“{text}”")
