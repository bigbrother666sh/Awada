import os
import pprint

from paddlenlp import Taskflow
from typing import (
    Optional, Union
)

from wechaty import (
    MessageType,
    WechatyPlugin,
    Message,
    WechatyPluginOptions
)
from wechaty_puppet import get_logger


class UieTestPlugin(WechatyPlugin):
    """
    基于Pyhton-wechat框架的AI soul实现
    用于Awada长期运营
    Author：bigbrother666
    All rights reserved 2022
    """
    def __init__(
            self,
            options: Optional[WechatyPluginOptions] = None,
    ) -> None:

        super().__init__(options)
        # 1. create the cache_dir
        self.cache_dir = f'./.{self.name}'
        self.file_cache_dir = f'{self.cache_dir}/file'
        os.makedirs(self.file_cache_dir, exist_ok=True)

        # 2. save the log info into <plugin_name>.log file
        log_file = os.path.join(self.cache_dir, 'log.log')
        self.logger = get_logger(self.name, log_file)

        # 3. check and load metadata
        schema = [{'作品': ['名称', '作者', '[看过，没看过]', '情感倾向[正向，负向]']}, {'作者': ['名字', '作品', '[喜欢，不喜欢]']}, {'角色': ['名字', '作品', '评价']}]

        try:
            self.uie = Taskflow('information_extraction', schema=schema)
        except Exception as e:
            self.logger.error('load uie failed, pls check the uie/checkpoint/model_best, be sure right model files exits')
            raise e

    async def on_message(self, msg: Message) -> None:
        talker = msg.talker()

        # 1. 判断是否是自己发送的消息\weixin service\room message
        if talker.contact_id != msg.is_self():
            return

        if msg.type() not in [MessageType.MESSAGE_TYPE_TEXT, MessageType.MESSAGE_TYPE_EMOTICON]:
            return
        text = msg.text()

        text = text.strip().replace('\n', '，')
        self.logger.info(f"processed text:{text}")

        nlu = self.uie(text)
        self.logger.info(f"nlu:{nlu}")
        pprint(nlu)
