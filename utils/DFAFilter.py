'''
原作Author guo
https://github.com/guojia60180/sensitive-words-filter
改写 bigbrother666
增加了测试代码，来自https://blog.csdn.net/u013421629/article/details/83178970
更改了返回模式，返回是否检测到和检测到的敏感词(检测到第一个敏感词就返回）
关键词数据来自：https://github.com/fwwdn/sensitive-stop-words
'''
import os
import logging


class DFAFilter:
    '''有穷状态机完成'''

    def __init__(self, logs: str = '.utils'):
        # 1. create the cache_dir
        self.cache_dir = logs
        os.makedirs(self.cache_dir, exist_ok=True)

        # 2. save the log info into <plugin_name>.log file
        self.logger = logging.getLogger('DAFFilter')
        self.logger.handlers = []
        self.logger.setLevel('INFO')
        self.logger.propagate = False
        log_file = os.path.join(self.cache_dir, 'daffilter_LTE.log')

        file_handler = logging.FileHandler(log_file, 'a', encoding='utf-8')
        file_handler.setLevel('INFO')
        self.logger.addHandler(file_handler)

        self.keywords_chains = {}
        self.delimit = '\x00'

    def add(self, keyword):
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return

        level = self.keywords_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]

            else:
                if not isinstance(level, dict):
                    break

                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level=level[chars[j]]

                last_level[last_char] = {self.delimit: 0}
                break

        if i == len(chars)-1:
            level[self.delimit] = 0

    file = os.path.split(os.path.realpath(__file__))[0]

    def parse(self, path=os.path.join(file, 'keywords')):
        with open(path, encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message):
        message = message.lower()
        start = 0
        while start < len(message):
            res = []
            level = self.keywords_chains
            for char in message[start:]:
                if char in level:
                    if self.delimit not in level[char]:
                        level = level[char]
                        res.append(char)
                    else:
                        res.append(char)
                        self.logger.info(f"文本：{message}'，检测到敏感词：{''.join(res)}")
                        return ''.join(res)
            start += 1
        return None


if __name__ == "__main__":
    import time
    time1 = time.time()
    gfw = DFAFilter()
    gfw.parse()
    print("====敏感词测试====")

    while True:
        print("输入Q退出")
        prompt = input("输入待测文本：")
        if prompt.lower() == "q":
            break
        result = gfw.filter(prompt)
        if result:
            print("检测出敏感词：", result)
        else:
            print("未查出敏感词")
        time2 = time.time()
        print('总共耗时：' + str(time2 - time1) + 's')
