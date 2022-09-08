from typing import Optional
from wechaty import (
    MessageType,
    WechatyPlugin,
    Message,
    WechatyPluginOptions
)
# from utils.DFAFilter import DFAFilter
from utils.rasaintent import RasaIntent


class Lurker(WechatyPlugin):
    """
    collect the data for DFA and rasa LTR
    """
    def __init__(self, options: Optional[WechatyPluginOptions] = None):
        super().__init__(options)

        self.intent = RasaIntent()

    async def on_message(self, msg: Message) -> None:
        # 1. 判断是否是自己发送的消息\weixin service\room message
        talker = msg.talker()
        if talker.contact_id == msg.is_self() or talker.contact_id == "weixin":
            return

        if msg.type() != MessageType.MESSAGE_TYPE_TEXT:
            return

        text = await msg.mention_text() if msg.room() else msg.text()
        if not text:
            return

        intent, cofidence = self.intent.predict(text)
        await msg.say(f"text:{text}, intent: {intent}, cofidence: {cofidence}")
