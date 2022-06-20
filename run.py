import asyncio
import os

from wechaty import Wechaty, WechatyOptions
from plugins.drama import DramaPlugin
from plugins.drama_zeus import DramaZeusPlugin

if __name__ == "__main__":
    options = WechatyOptions(
        port=int(os.environ.get('PORT', 8004)),
    )
    bot = Wechaty(options)
    bot.use([
        DramaPlugin(),
        DramaZeusPlugin(24.7dc77c87ce71ed73b73d714728672313.86400000.1655779572205.e874c242e1a293ad1972230edbf69e42-36698)
    ])
    asyncio.run(bot.start())
