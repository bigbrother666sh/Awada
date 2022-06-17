import asyncio
import os

from wechaty import Wechaty, WechatyOptions
from plugins.drama import DramaPlugin


if __name__ == "__main__":
    options = WechatyOptions(
        port=int(os.environ.get('PORT', 8004)),
    )
    bot = Wechaty(options)
    bot.use([
        DramaPlugin()
    ])
    asyncio.run(bot.start())
