import os
import json
import urllib3
import time

import wechaty
from paddlenlp import Taskflow
from typing import (
    Optional, Union
)
import xlrd
from wechaty import (
    Contact,
    MessageType,
    WechatyPlugin,
    Message,
    WechatyPluginOptions
)
from wechaty_puppet import get_logger
from utils.DFAFilter import DFAFilter
from plugins.inspurai.inspurai import Yuan


class DramaPlugin(WechatyPlugin):
    """
    基于Pyhton-wechat框架的AI soul实现
    用于Awada长期运营
    Author：bigbrother666
    All rights reserved 2022
    """
    def __init__(
            self,
            options: Optional[WechatyPluginOptions] = None,
            configs: str = 'drama_configs',
            port: str = '5005'
    ) -> None:

        super().__init__(options)
        # 1. create the cache_dir
        self.config_url = configs
        self.config_files = os.listdir(self.config_url)
        if len(self.config_files) < 5:
            raise RuntimeError('Drada plugin config_files not enough, pls add and try again')

        self.cache_dir = f'./.{self.name}'
        self.file_cache_dir = f'{self.cache_dir}/file'
        os.makedirs(self.file_cache_dir, exist_ok=True)

        # 2. save the log info into <plugin_name>.log file
        log_file = os.path.join(self.cache_dir, 'log.log')
        self.logger = get_logger(self.name, log_file)

        # 3. check and load metadata
        if self._file_check() is False:
            raise RuntimeError('Drada plugin needs above config_files, pls add and try again')

        with open(os.path.join(self.config_url, 'directors.json'), 'r', encoding='utf-8') as f:
            self.directors = json.load(f)
        if len(self.directors) == 0:
            self.logger.warning('there must be at least one director, pls retry')
            raise RuntimeError('Drama director.json not valid, pls refer to above info and try again')

        self.mmrules = self._load_mmrules()
        if self.mmrules is None:
            raise RuntimeError('Drada MMrules.xlsx not valid, pls refer to above info and try again')

        # 4. load self-memory data and create memory-dict for users
        with open(os.path.join(self.config_url, 'focus.json'), 'r', encoding='utf-8') as f:
            schema = json.load(f)
        if len(schema) == 0 or "" in schema:
            self.logger.warning('there must be at least one in the focus.json and no empty should be, pls retry')
            raise RuntimeError('Drama focus.json not valid, pls refer to above info and try again')

        schema.extend(['时间', '地点', '人物'])
        schema = list(set(schema))

        try:
            self.uie = Taskflow('information_extraction', schema=schema, task_path='uie/checkpoint/model_best')
        except Exception as e:
            self.logger.error('load uie failed, pls check the uie/checkpoint/model_best, be sure right model files exits')
            raise e

        self.self_memory = self._load_memory()
        if self.self_memory is None:
            raise RuntimeError('Drada memory.txt not valid, pls refer to above info and try again')

        if "user_memory.json" in self.config_files:
            with open(os.path.join(self.config_url, 'user_memory.json'), 'r', encoding='utf-8') as f:
                self.user_memory = json.load(f)
        else:
            self.user_memory = {}

        if "users.json" in self.config_files:
            with open(os.path.join(self.config_url, 'users.json'), 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}

        # 5. load scenario rule-table
        self.scenarios, self.last_turn_memory_template = self._load_scenarios()
        if self.scenarios is None:
            raise RuntimeError('Drada scenarios.xlsx not valid, pls refer to above info and try again. make sure at lease one scenario is well defined.')

        self.last_turn_memory = {}
        if self.user_memory:
            for key in self.user_memory.keys():
                self.last_turn_memory[key] = self.last_turn_memory_template

        # 6. initialize & test the rasa nlu server and yuan-api
        self.rasa_url = 'http://localhost:'+port+'/model/parse'
        self.http = urllib3.PoolManager()

        _test_data = {'text': '苍老师德艺双馨'}
        _encoded_data = json.dumps(_test_data)
        _test_res = self.http.request('POST', self.rasa_url, body=_encoded_data)
        _result = json.loads(_test_res.data)

        if not _result:
            raise RuntimeError('Rasa server not running, pls start it first and trans the right port in str')

        self.yuan = Yuan(engine='dialog',
                    input_prefix="",
                    input_suffix="",
                    output_prefix="",
                    output_suffix="",)

        engine_name = self.yuan.get_engine()
        self.logger.info(f'with yuan engine:{engine_name},with temperature=0.9,max_tokens=100,topK=1,topP=0.9')

        # 7. last process
        self.gfw = DFAFilter()
        self.gfw.parse()

        self.take_over = False
        self.temp_talker: wechaty.Contact
        self.take_over_director: wechaty.Contact
        self.last_text = {}

        self.logger.info('Drada plugin init success')

    def _file_check(self) -> bool:
        """check the config file"""

        if "directors.json" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have directors.json!')
            return False

        if "focus.json" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have focus.json!')
            return False

        if "MMrules.xlsx" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have MMrules.xlsx!')
            return False

        if "memory.txt" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have memory.txt!')
            return False

        if "scenarios.xlsx" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have scenarios.xlsx!')
            return False

    def _load_memory(self) -> None or dict:
        """load the memory data"""
        memory_file = os.path.join(self.config_url, 'memory.txt')
        with open(memory_file, 'r', encoding='utf-8') as f:
            datas = [line.strip() for line in f.readlines() if line.strip()]

        if len(datas) == 0:
            self.logger.warning('no data in memory.txt,this is not allowed')
            return None

        focus = self.nlu_info(datas)

        self_memory = {}
        for i in range(len(datas)):
            for entity in focus[i]:
                if entity in self_memory.keys():
                    self_memory[entity].append(datas[i])
                else:
                    self_memory[entity] = [datas[i]]

        return self_memory

    def _load_mmrules(self) -> None or dict:
        """load the Memory Mathmatics Rules from excel"""
        mmrules = os.path.join(self.config_url, 'MMrules.xlsx')
        data = xlrd.open_workbook(mmrules)
        table = data.sheets()[0]

        nrows = table.nrows
        if nrows == 0:
            self.logger.warning('no memory in MMrules.xls,this is not allowed')
            return None

        cols = table.ncols

        if table.cell_value(0,1).lower() != 'read' or table.cell_value(0,2).lower() != 'bi':
            self.logger.warning('MMrules.xlsx is not in the right format:column 1 and 2 must be read and bi')
            return None

        rules = {}
        for i in range(1, nrows):
            for k in range(cols):
                if k == 0:
                    if table.cell_value(i,k):
                        intent = table.cell_value(i,k)
                        rules[intent] = {}
                    else:
                        self.logger.warning('MMrules.xlsx is not in the right format: intent is empty')
                        return None
                    continue
                if table.cell_value(i,k).lower() not in ['yes','no']:
                    self.logger.warning('MMrules.xlsx is not in the right format: value is not yes or no')
                    return None
                else:
                    rules[intent][table.cell_value(0,k).lower()] = table.cell_value(i,k).lower()

        return rules

    def _load_scenarios(self) -> None or dict:
        """load the scenarios data"""
        scenarios_file = os.path.join(self.config_url, 'scenarios.xlsx')
        data = xlrd.open_workbook(scenarios_file)

        rules = {}
        last_turn_memory_template = {}
        for name in data.sheet_names():
            table = data.sheet_by_name(name)
            nrows = table.nrows
            if nrows < 2:
                continue

            for i in range(1, nrows):
                if not table.cell_value(i,0):
                    self.logger.warning('cell of the first column is empty in scenario.xlsx,this is not allowed')
                    return None, None

            cols = table.ncols
            for k in range(1, cols):
                if not table.cell_value(0,k):
                    self.logger.warning('cell of the first row is empty in scenario.xlsx,this is not allowed')
                    return None, None

            rules[name] = {}
            last_turn_memory_template[name] = {}
            for k in range(1, cols):
                last_turn_memory_template[name][table.cell_value(0, k)] = []
                rules[name][table.cell_value(0, k)] = {}
                for i in range(1, nrows):
                    if table.cell_value(i,k):
                        rules[name][table.cell_value(0, k)][table.cell_value(i, 0)] = table.cell_value(i, k)

        return rules, last_turn_memory_template


    async def director_message(self, msg: Message):
        """
        Director Module
        the multy-media-message would be added in next stage
        """
        # 1. check the heartbeat of DramaPlugin
        if msg.text() == "ding":
            await msg.say('dong -- DramaPlugin')
            return
        # 2. help menu
        if msg.text() == 'help':
            await msg.say("Drama Director Code: \n"
                          "ding -- check heartbeat \n"
                          "reload directors --- reload director.json \n"
                          "reload mmrules -- reload MMrules.xlsx \n"
                          "reload memory -- reload memory.txt \n"
                          "reload scenarios -- reload scenarios.xlsx \n"
                          "save -- save the users status and users memory so that game will continue instead of restart\n"
                          "take over -- take over the AI for a time \n"
                          "take over off -- stop the take_over")
            return
        # 3.functions
        if msg.text() == 'reload directors':
            with open(os.path.join(self.config_url, 'directors.json'), 'r', encoding='utf-8') as f:
                directors = json.load(f)
            if len(directors) == 0:
                await msg.say('there must be at least one director, director list not changed')
            else:
                self.directors = directors
                await msg.say('Drama director list has been updated')
            return

        if msg.text() == 'reload mmrules':
            mmrules = self._load_mmrules()
            if mmrules is None:
                await msg.say("Drada MMrules.xlsx not valid, I'll keep the old set. no change happened")
            else:
                self.mmrules = mmrules
                await msg.say("Drada MMrules has been updated")
            return

        if msg.text() == 'reload memory':
            selfmemory = self._load_memory()
            if selfmemory is None:
                await msg.say("memory.txt is empty, so I will not change my memory")
            else:
                self.self_memory = selfmemory
                await msg.say("self memory has been updated.")
            return

        if msg.text() == 'reload scenarios':
            scenarios, last_turn_memory_template = self._load_scenarios()
            if scenarios is None:
                await msg.say("scenarios.xlsx is empty, so I will not reload scenarios. No change happened")
            elif last_turn_memory_template.keys() != self.last_turn_memory_template.keys():
                await msg.say("You should not change peoples in the scenarios during program is running, you have to restart to update")
            else:
                self.scenarios = scenarios
                self.last_turn_memory_template = last_turn_memory_template
                await msg.say("scenarios has been updated")
            return

        if msg.text() == 'save':
            with open(os.path.join(self.config_url, 'users.json'), 'w', encoding='utf-8') as f:
                json.dump(self.users, f)
            with open(os.path.join(self.config_url, 'user_memory.json'), 'w', encoding='utf-8') as f:
                json.dump(self.user_memory, f)
            # for director may change the people of scenarios during restart, so we donot save the last_turn_memory
            await msg.say(f"user status and memory has been saved in {self.config_url}. I'll read instead of create new till you delete the files")
            return

        if msg.text() == "take over":
            self.take_over = True
            self.take_over_director = await self.bot.Contact.find(msg.talker().name)
            await msg.say("ok your turn. to give the wheel back to me send: take over off")
            return

        if msg.text() == 'take over off':
            self.take_over = False
            await msg.say("I will take the talk again. to take over send: take over")
            return

        if self.take_over:
            await msg.forward(self.temp_talker)
            await msg.say(f"msg has been forward to {self.temp_talker.name}")
            self.last_turn_memory[self.temp_talker.contact_id][self.users[self.temp_talker.contact_id][1]][self.users[self.temp_talker.contact_id][0]].append(f'你说：“{msg.text()}”')
        else:
            await msg.say("send help to me to check what you can do")

    def nlu_intent(self, text: str) -> str:
        _test_data = {'text': text}
        _encoded_data = json.dumps(_test_data)
        _test_res = self.http.request('POST', self.rasa_url, body=_encoded_data)
        _result = json.loads(_test_res.data)
        return _result['intent']['name']

    def nlu_info(self, text:Union[str,list]) -> list:
        results = self.uie(text)
        infos = []
        for _result in results:
            info = []
            for result in _result.values():
                for entity in result:
                    if entity['probability'] > 0.64:
                        info.append(entity['text'])
            infos.append(list(set(info)))
        return infos

    async def soul(self, text: str, talker: Contact, scenario: str, character: str, memory: dict, last_dialog: list, rules:dict) -> None:
        # 1. understanding: intent judgment, focus information_extraction
        intent = self.nlu_intent(text)
        infos = self.nlu_info([self.last_text[talker.contact_id], text])
        info = list(set(infos[0]+infos[1]))

        self.logger.info(f"intent:{intent}")
        self.logger.info(f"entities:{','.join(info)}")

        # 2. memory reading
        pre_prompt = rules['DESCRIPTION'] + '。'
        memory_text = ''
        if info:
            memory_squence = []
            for entity in info:
                memory_squence += memory.get(entity, [])

            if memory_squence:
                memory_squence.sort(key=lambda k: (k.get('time')), reverse=False)
                for sentence in memory_squence:
                    if sentence['text'] not in last_dialog and sentence['text'] not in memory_text:
                        memory_text += sentence['text']

        pre_prompt = pre_prompt + memory_text + ''.join(last_dialog) + character + "说：“" + text + "”"

        if info and self.mmrules[intent]['read'] == 'yes':
            selfmemory_squence = []
            for entity in info:
                selfmemory_squence += self.self_memory.get(entity, [])
            selfmemory_text = ''.join(set(selfmemory_squence))
            pre_prompt = pre_prompt + "，你知道" + selfmemory_text + "于是"

        # 3. saving user's text as memory according to entity
        t = time.time()
        for entity in info:
            if entity in self.user_memory[talker.contact_id].keys():
                self.user_memory[talker.contact_id][entity].append({"time": t, "text": f'{character}说：“{text}”'})
            else:
                self.user_memory[talker.contact_id][entity] = [{"time": t, "text": f'{character}说：“{text}”'}]

        # 4. act the action in sequence
        addtion_action = ''
        for entity in info:
            if entity in rules.keys():
                addtion_action = rules[entity]
                break

        pre_prompt = pre_prompt + "你" + addtion_action

        actions = rules.get(intent, '').split('\n')

        replies = ''
        for action in actions:
            if action.startswith('S'):
                reply = action[1:]
            else:
                prompt = pre_prompt + action + "说：“"
                self.logger.info(prompt)
                self.logger.info("----------------------------\n")
                reply = self.yuan.submit_API(prompt, trun="”")
                if reply is None:
                    self.logger.warning(f'generation failed with the following input:{character},{intent},{action},{text},{scenario}')
                    continue
                if reply == "somethingwentwrongwithyuanservice":
                    self.logger.warning(f'Yuan is out of service, failed from:{character},{talker.name},{text},{scenario}')
                    continue
            await talker.say(reply)
            replies += f'你说：“{reply}”'

        self.last_turn_memory[talker.contact_id][scenario][character].append(replies)

        # 3. memory saving acording to MMrules
        if self.mmrules[intent]['bi'] == 'no':
            return

        time.sleep(0.1)
        t = time.time()
        for entity in info:
            if entity in self.user_memory[talker.contact_id].keys():
                self.user_memory[talker.contact_id][entity].append({"time": t, "text": replies})
            else:
                self.user_memory[talker.contact_id][entity] = [{"time": t, "text": replies}]

    async def on_message(self, msg: Message) -> None:
        talker = msg.talker()

        # 1. 判断是否是自己发送的消息\weixin service\room message
        if talker.contact_id == msg.is_self() or talker.contact_id == "weixin" or msg.room():
            return

        # 2. check if is director
        if talker.contact_id in self.directors:
            await self.director_message(msg)
            return

        # 3. new-user register and old-user session load
        """
        A storyline mechanism should be here to judge and guide the transition between scenarios, which I will do in the future
        welcome should also be a scenario
        and there should be something program can learn when and how to bring user from one scenario to another
        """
        if talker.contact_id not in self.users.keys():
            self.users[talker.contact_id] = ['张无忌', 'welcome']
            self.user_memory[talker.contact_id] = {}
            self.last_turn_memory[talker.contact_id] = self.last_turn_memory_template
            self.last_text[talker.contact_id] = '呵呵'
            #实际上这里是用talker.contact_id充当"局"的概念，即同样的游戏可能同时开好几局，所有的背景记忆是一样的，但是用户相关的记忆是要分开的。
            #因为这一次是单场景单角色，所以就相当于"一个用户是一句"了，所以用contact_id作为区分，假如是剧本啥这种，就可以用room_id
            await talker.say("先声明哈，我们之间的对话信息可能会被公开，介意的话请终止对话！\n"
                             "请您务必不要透露任何隐私信息，请您务发表不当言论")
            return

        # 4. message pre-process
        """
        1. 是否是文本消息，排除不支持的消息类型（目前只支持文本，另外支持一个emoj，emoj统一识别为：嘿嘿）
        2. 敏感词检测
        3. 去掉特殊符号，比如@ \n等
        """
        if msg.type() not in [MessageType.MESSAGE_TYPE_TEXT, MessageType.MESSAGE_TYPE_EMOTICON]:
            return

        if msg.type() == MessageType.MESSAGE_TYPE_EMOTICON:
            text = '嘿嘿'
        else:
            text = msg.text()

        text = text.strip().replace('\n', '，')
        self.logger.info(f"processed text:{text}")

        if self.gfw.filter(text):
            self.logger.info(f'{text} is filtered, for the reason of {self.gfw.filter(text)}')
            await msg.say('请勿发表不当言论，谢谢配合')
            return

        # 5. check the status of the talker. for special status do the special action
        scenario = self.users[talker.contact_id][1]
        character = self.users[talker.contact_id][0]
        memory = self.user_memory[talker.contact_id]
        last_dialog = self.last_turn_memory[talker.contact_id][scenario][character]
        rules = self.scenarios[scenario].get(character, {'DESCRIPTION':''})
        self.temp_talker = talker

        if scenario == 'welcome':
            await talker.say("just the same as description in scenario yitiantulong")
            self.users[talker.contact_id][1] = "yitiantulong"
            return

        self.last_turn_memory[talker.contact_id][scenario][character] = [f'{character}说：“{text}”']
        if self.take_over is True:
            await self.take_over_director.say(f"{character} in the {scenario} just say: {text}. pls reply directly here")
            await self.take_over_director.say(f"last turn dialog: {''.join(last_dialog)}")
        else:
            await self.soul(text, talker, scenario, character, memory, last_dialog, rules)

        self.last_text[talker.contact_id] = text

        # colorful eggs https://ai.baidu.com/ai-doc/wenxin/Zl33wtflg
