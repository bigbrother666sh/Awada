import os
import json
from typing import Optional
from wechaty import (
    MessageType,
    WechatyPlugin,
    Message,
    WechatyPluginOptions
)
from utils.DFAFilter import DFAFilter
from utils.rasaintent import RasaIntent


class Lurker(WechatyPlugin):
    """
    collect the data for DFA and rasa LTR
    """
    def __init__(self, options: Optional[WechatyPluginOptions] = None, configs: str = 'CAconfigs'):
        super().__init__(options)
        # 1. init the config file
        self.config_url = configs
        self.config_files = os.listdir(self.config_url)

        with open(os.path.join(self.config_url, 'rooms.json'), 'r', encoding='utf-8') as f:
            self.rooms = json.load(f)

        self.gfw = DFAFilter()
        self.gfw.parse()
        self.intent = RasaIntent()

    async def on_message(self, msg: Message) -> None:
        # 1. 判断是否是自己发送的消息\weixin service\room message
        talker = msg.talker()
        if talker.contact_id == msg.is_self() or talker.contact_id == "weixin":
            return

        if msg.type() != MessageType.MESSAGE_TYPE_TEXT:
            return

        # 3. rooms
        if msg.room() and msg.room().room_id in self.rooms:
            self.gfw.filter(msg.text())
            self.intent.predict(msg.text())
