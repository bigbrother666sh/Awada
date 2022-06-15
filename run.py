import asyncio
import os

from wechaty import Wechaty, WechatyOptions
#from plugins.drama import DramaPlugin
from plugins.test1 import DramaPluginTest
from plugins.test2 import CAPluginTest


if __name__ == "__main__":
    options = WechatyOptions(
        port=int(os.environ.get('PORT', 8004)),
    )
    bot = Wechaty(options)
    bot.use([
        DramaPluginTest(),
        CAPluginTest()
    ])
    asyncio.run(bot.start())
