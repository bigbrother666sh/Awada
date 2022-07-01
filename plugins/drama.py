import os
import json
import time
import re
import wechaty
from paddlenlp import Taskflow
from typing import Optional
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
from utils.rasaintent import RasaIntent
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
    ) -> None:

        super().__init__(options)
        # 1. create the cache_dir
        self.config_url = configs
        self.config_files = os.listdir(self.config_url)
        if len(self.config_files) < 4:
            raise RuntimeError('Drada plugin config_files not enough, pls add and try again')

        self.cache_dir = f'./.{self.name}'
        self.file_cache_dir = f'{self.cache_dir}/file'
        os.makedirs(self.file_cache_dir, exist_ok=True)

        # 2. save the log info into <plugin_name>.log file
        log_file = os.path.join(self.cache_dir, 'log.log')
        self.logger = get_logger(self.name, log_file)

        # 3. check and load directors
        if self._file_check() is False:
            raise RuntimeError('Drada plugin needs above config_files, pls add and try again')

        with open(os.path.join(self.config_url, 'directors.json'), 'r', encoding='utf-8') as f:
            self.directors = json.load(f)
        if not self.directors or "" in self.directors:
            self.logger.warning('there must be at least one director and no null items, pls retry')
            raise RuntimeError('Drama director.json not valid, pls refer to above info and try again')

        # 4. load scenario rule-table
        self.scenarios = self._load_scenarios()
        if self.scenarios is None:
            raise RuntimeError('Drada scenarios.xlsx not valid, pls refer to above info and try again. make sure at lease one scenario is well defined.')

        # 5. load self-memory systems: self_memory\relations\user_memory\last_turn_memory
        with open(os.path.join(self.config_url, 'focus.json'), 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        if not self.schema or "" in self.schema:
            self.logger.warning('there must be at least one in the focus.json and no empty should be, pls retry')
            raise RuntimeError('Drama focus.json not valid, pls refer to above info and try again')

        self.uie = Taskflow('information_extraction', schema=self.schema, task_path='uie/checkpoint/model_best')

        self.self_memory = self._load_memory()
        if not self.self_memory:
            raise RuntimeError('Drada memory.txt not valid, pls refer to above info and try again')

        if "relations.txt" in self.config_files:
            self.relations = self._load_relations()
        else:
            self.relations = {}

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

        if "last_turn_memory.json" in self.config_files:
            with open(os.path.join(self.config_url, 'last_turn_memory.json'), 'r', encoding='utf-8') as f:
                self.last_turn_memory = json.load(f)
        else:
            self.last_turn_memory = {}
            if self.user_memory:
                with open(os.path.join(self.file_cache_dir, 'last_turn_memory_template.json'), 'r', encoding='utf-8') as f:
                    for key in self.user_memory.keys():
                        self.last_turn_memory[key] = json.load(f)

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

        engine_name = self.yuan.get_engine()
        self.logger.info(f'with yuan engine:{engine_name},with temperature=1, max_tokens=150, topK=3, topP=0.9, frequencyPenalty=1.2')

        # 7. last process
        self.gfw = DFAFilter()
        self.gfw.parse()
        self.intent = RasaIntent()

        self.take_over = False
        self.temp_talker: wechaty.Contact
        self.take_over_director: wechaty.Contact

        self.logger.info(f'Drada plugin init success.')

    def _file_check(self) -> bool:
        """check the config file"""

        if "directors.json" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have directors.json!')
            return False

        if "focus.json" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have focus.json!')
            return False

        if "memory.txt" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have memory.txt!')
            return False

        if "scenarios.xlsx" not in self.config_files:
            self.logger.warning(f'config file url:/{self.config_url} does not have scenarios.xlsx!')
            return False

        if "relations.txt" not in self.config_files:
            self.logger.warning(f"config file url:/{self.config_url} does not have relations.txt. however you can go without it, but it's not recommended!")

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
            self.logger.warning('no data in memory.txt,this is not allowed')
            return self_memory_text

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
            self.logger.warning('no data in relations.txt or the number of lines is not even')
            return {}

        return dict(zip(relations_text[::2], relations_text[1::2]))

    """
    def _load_mmrules(self) -> None or dict:
        #load the Memory Mathmatics Rules from excel
        #deprecation for this stage
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
    """

    def _load_scenarios(self) -> dict:
        """load the scenarios data"""
        scenarios_file = os.path.join(self.config_url, 'scenarios.xlsx')
        data = xlrd.open_workbook(scenarios_file)

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

        with open(os.path.join(self.file_cache_dir, 'last_turn_memory_template.json'), 'w', encoding='utf-8') as f:
            json.dump(last_turn_memory_template, f, ensure_ascii=False)

        return rules

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
                          "reload memory -- reload memory.txt \n"
                          "reload scenarios -- reload scenarios.xlsx \n"
                          "add focus文本 -- add new focus text(xx)\n"
                          "add selfmemory文本 -- add new self memory text(xx)\n"
                          "save -- save the users status and users memory so that game will continue instead of restart\n"
                          "take over -- take over the AI for a time \n"
                          "stop take over -- stop the take_over")
            return
        # 3.functions
        if msg.text().startswith('reload directors'):
            with open(os.path.join(self.config_url, 'directors.json'), 'r', encoding='utf-8') as f:
                directors = json.load(f)
            if len(directors) == 0:
                await msg.say('there must be at least one director, director list not changed')
            else:
                self.directors = directors
                await msg.say('Drama director list has been updated')
            return

        if msg.text().startswith('add selfmemory'):
            if msg.text()[14:] is None:
                await msg.say("add the focus text close to the code, pls try again")
                return
            topics = self.nlu_topic([msg.text()[14:]])
            self.self_memory.append({**{"text": msg.text()[14:]}, **topics[0]})
            with open(os.path.join(self.config_url, 'memory.txt'), 'a', encoding='utf-8') as f:
                f.write(msg.text()[14:] + '\n')
            await msg.say(f"self_memory added new item:{msg.text()[14:]} and memory.txt has been auto updated")
            return

        if msg.text().startswith('add focus'):
            if msg.text()[9:] is None:
                await msg.say("add the focus text close to the code, pls try again")
                return
            self.schema.append(msg.text()[9:])
            self.uie.set_schema(self.schema)
            self.self_memory = self._load_memory()
            with open(os.path.join(self.config_url, 'focus.json'), 'w', encoding='utf-8') as f:
                json.dump(self.schema, f, ensure_ascii=False)
            await msg.say("focus updated, and self_memory has been updated. focus.json has been auto updated")
            return

        if msg.text().startswith('reload memory'):
            selfmemory = self._load_memory()
            if not selfmemory:
                await msg.say("memory.txt is empty, so I will not change my memory")
            else:
                self.self_memory = selfmemory
                await msg.say("self memory has been updated.")
            return

        if msg.text().startswith('reload scenarios'):
            scenarios = self._load_scenarios()
            if scenarios is None:
                await msg.say("scenarios.xlsx is empty, so I will not reload scenarios. No change happened")
            else:
                self.scenarios = scenarios
                await msg.say("warning: any change of scenarios or characters during program running may cause potentially fatal error！")
                await msg.say("scenarios has been updated")
            return

        if msg.text().startswith('save'):
            with open(os.path.join(self.config_url, 'users.json'), 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False)
            with open(os.path.join(self.config_url, 'last_turn_memory.json'), 'w', encoding='utf-8') as f:
                json.dump(self.last_turn_memory, f, ensure_ascii=False)
            await msg.say("user status and last turn memory has been saved. I'll read instead of create new till you delete the files \n"
                          "Attention: You Should not change the characters of scenarios untill you remove the user_memory.json and last_turn_memory.json")
            return

        if msg.text().startswith("take over"):
            self.take_over = True
            self.take_over_director = await self.bot.Contact.find(msg.talker().name)
            await msg.say("ok your turn. to give the wheel back to me send: take over off")
            return

        if msg.text().startswith('stop take over'):
            self.take_over = False
            await msg.say("I will take the talk again. to take over send: take over")
            return

        if self.take_over:
            await msg.forward(self.temp_talker)
            await msg.say(f"msg has been forward to {self.temp_talker.name}")
            self.last_turn_memory[self.temp_talker.contact_id][self.users[self.temp_talker.contact_id][1]][self.users[self.temp_talker.contact_id][0]]["text"] = [f'你说：“{msg.text()}”']
            self.last_turn_memory[self.temp_talker.contact_id][self.users[self.temp_talker.contact_id][1]][self.users[self.temp_talker.contact_id][0]]["talker"] = ['你']
        else:
            await msg.say("send help to me to check what you can do")

    async def soul(self, text: str, talker: Contact, scenario: str, character: str, memory: list, last_dialog: str, rules: dict) -> None:
        # 1. understanding: topic information_extraction
        topics = self.nlu_topic([text, last_dialog])
        if topics[0]:
            topic = topics[0]
        else:
            topic = topics[1]
        self.logger.info(f"topics:{topic}")

        # 2. memory reading
        selfmemory_text = ''
        if topic:
            for shcema in self.schema:
                if not topic.get(shcema, []):
                    continue
                for _memory in self.self_memory:
                    if topic[shcema].intersection(_memory.get(shcema, set([]))):
                        selfmemory_text += _memory['text']
                if selfmemory_text:
                    break

        memory_text = ''
        if topic:
            for shcema in self.schema:
                if not topic.get(shcema, []):
                    continue
                for _memory in memory:
                    if topic[shcema].intersection(_memory.get(shcema, set([]))):
                        memory_text += _memory['text']
                if memory_text:
                    break

        # 3. combine to the pre_prompt
        if memory_text:
            pre_prompt = self.relations.get("你", "") + self.relations.get(character, '') + selfmemory_text + rules.get('DESCRIPTIONTEXT', '') + '刚才' + memory_text + '，现在' + last_dialog + "你"
        else:
            pre_prompt = self.relations.get("你", "") + self.relations.get(character, '') + selfmemory_text + rules.get('DESCRIPTIONTEXT', '') + last_dialog + "你"

        # 4. act the action in sequence
        actions = []
        if topic:
            for shcema in self.schema:
                if not topic.get(shcema, []):
                    continue
                for ner in list(topic[shcema]):
                    actions.extend(rules.get(ner, '').split('\n'))
                if actions:
                    break

        if not actions:
            actions = ['']

        replies = []
        for action in actions:
            if action.startswith('SOLID'):
                reply = action[5:]
            elif action.startswith('TRANS'):
                self.users[talker.contact_id][1] = action[5:]
                next_rules = self.scenarios[action[5:]].get(character, list(self.scenarios[action[5:]].values())[0])
                if 'WELCOMEWORD' in next_rules:
                    await talker.say(next_rules['WELCOMEWORD'])
                    self.last_turn_memory[talker.contact_id][action[5:]][character]["text"] = [f"你说：“{next_rules['WELCOMEWORD']}”"]
                    self.last_turn_memory[talker.contact_id][action[5:]][character]["talker"] = ["你"]
                    if topic:
                        memory.append({**{"text": last_dialog}, **topic})
                return
            elif action.startswith('HOLD'):
                try:
                    time.sleep(int(action[4:]))
                    self.logger.info(f"HOlD {action[4:]}s as editor wish")
                except ValueError:
                    self.logger.info(f"may the action format wrong, scenario:{scenario}, characters:{character}, action: {action}")
                continue
            elif action.startswith('SILENCE'):
                self.logger.info(f"not reply as editor wish")
                continue
            else:
                prompt = pre_prompt + action + "说：“"
                self.logger.info(prompt)
                for i in range(7):
                    reply = self.yuan.submit_API(prompt, trun="”")
                    if reply == '':
                        self.logger.warning(f'generation failed {str(i+1)} times.')
                        continue
                    if len(reply) <= 5 or reply not in last_dialog:
                        break

                if reply == '' or reply == "somethingwentwrongwithyuanservice":
                    self.logger.warning(f'Yuan may out of service')
                    continue

            await talker.say(reply)
            self.logger.info(f'你说：“{reply}”')
            replies.append(reply)

        self.logger.info("----------------------------\n")

        # 5. memory saving
        if topic:
            memory.append({**{"text": last_dialog}, **topic})
        self.last_turn_memory[talker.contact_id][scenario][character]["text"] = [f"你说：“{'。'.join(replies)}”"]
        self.last_turn_memory[talker.contact_id][scenario][character]["talker"] = ["你"]

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
        Now we have a first version of the storyline mechanism, 
        which can realize automatic switching between scenarios. 
        The code level only needs to define the initial state.
        """
        if talker.contact_id not in self.users:
            self.users[talker.contact_id] = ['陌生人', 'welcome']
            self.user_memory[talker.contact_id] = dict.fromkeys(self.scenarios.keys(), [])
            with open(os.path.join(self.file_cache_dir, 'last_turn_memory_template.json'), 'r', encoding='utf-8') as f:
                self.last_turn_memory[talker.contact_id] = json.load(f)
            """
            整体上应该是，一个剧本有很多局，每一局对应特定的一群玩家，每个玩家对应一个character【这些信息体现在users.json】
            然后剧本有很多scenario，每个scenario里面包含多个room，room为最小"对话窗口"，一个room可以含一个或多个character
            即 games>scenarios>rooms>characters
            last_turn_memory是以room为单位组织的
            但user_memory是以scenarios为单位组织的
            而对应不同的intent的action，我们称为rules，它是按scenarios-room组织的，但在scenario.xlxs表格中你可以直接写character，这就意味着这个房间只有AI和character，相当于私聊
            另外这个案例中，因为只是单场景、单character，所以我们直接用talkerid当做games的标记
            """
            await talker.say("先声明哈，我们之间的对话信息可能会被公开，介意的话请终止对话！\n"
                             "请您务必不要透露任何隐私信息，请您勿发表不当言论")
            if 'WELCOMEWORD' in self.scenarios['welcome'].get('陌生人', {}):
                await talker.say(self.scenarios['welcome']['陌生人']['WELCOMEWORD'])
                self.last_turn_memory[talker.contact_id]['welcome']['陌生人']["text"] = [f"你说：“{self.scenarios['welcome']['陌生人']['WELCOMEWORD']}”"]
                self.last_turn_memory[talker.contact_id]['welcome']['陌生人']["talker"] = ["你"]
            return

        # 4. message pre-process
        """
        1. 是否是文本消息，排除不支持的消息类型（目前只支持文本）
        2. 支持引用消息，会自动解析并拼接
        2. 敏感词检测
        3. 去掉特殊符号，比如@ \n等
        """
        if msg.type() != MessageType.MESSAGE_TYPE_TEXT:
            return

        text = msg.text()
        if re.match(r"^「.+」\s-+\s.+", text, re.S):
            text = re.search(r"：.+」", text, re.S).group()[1:-1] + "，" + re.search(r"-\n.+", text, re.S).group()[2:]

        text = text.strip().replace('\n', '，')
        self.logger.info(f"processed text:{text}")

        if self.gfw.filter(text):
            self.logger.info(f'{text} is filtered, for the reason of {self.gfw.filter(text)}')
            await msg.say('请勿发表不当言论，谢谢配合')
            return

        if self.users[talker.contact_id][1] == 'heddacomeagain':
            self.users[talker.contact_id][1] = 'welcome'
        elif self.users[talker.contact_id][1] == 'bye':
            return

        # 5. check the status of the talker. for special status do the special action
        scenario = self.users[talker.contact_id][1]
        character = self.users[talker.contact_id][0]
        if character == self.last_turn_memory[talker.contact_id][scenario][character]["talker"][-1]:
            self.last_turn_memory[talker.contact_id][scenario][character]["text"][-1] = self.last_turn_memory[talker.contact_id][scenario][character]["text"][-1][:-1] + '，' + f'{text}”'
        else:
            self.last_turn_memory[talker.contact_id][scenario][character]["text"].append(f'{character}说：“{text}”')

        self.last_turn_memory[talker.contact_id][scenario][character]["talker"].append(character)
        last_dialog = ''.join(self.last_turn_memory[talker.contact_id][scenario][character]["text"])

        # 6. check if director hand over
        self.temp_talker = talker
        if self.take_over is True:
            await self.take_over_director.say(f"{character} in the {scenario} just say: {text}. pls reply directly here")
            await self.take_over_director.say(f"whole turn dialog: {last_dialog}")
            return

        # 7. AI process
        rules = self.scenarios[scenario].get(character, {})
        memory = self.user_memory[talker.contact_id][scenario]
        intent = self.intent.predict(text)
        self.logger.info(f"intent:{intent}")
        """
        for Rasa cannot guarantee precision at this stage, we need to partially correct the intent,
        This part needs to be enriched
        if len(text) > 3:
            _i = 0
            for _memory in memory:
                if text in _memory["text"]:
                    _i += 1
                    if _i > 2:
                        intent = 'challenge'
        """
        if intent in ['notinterest', 'bye']:
            await msg.say('[微笑]')
        elif intent == 'praise':
            await msg.say('[666]记得要去看哦~')
        elif intent in ['challenge', 'challenge_bye']:
            await msg.say('呵呵……人类[困]')
        else:
            await self.soul(text, talker, scenario, character, memory, last_dialog, rules)
