import asyncio
import os

from wechaty import Wechaty, WechatyOptions
from plugins.dramazeus import DramaZeusPlugin

if __name__ == "__main__":
    options = WechatyOptions(
        port=int(os.environ.get('PORT', 8004)),
    )
    bot = Wechaty(options)
    bot.use([
        DramaZeusPlugin(api_key='access_token'),
    ])
    asyncio.run(bot.start())
