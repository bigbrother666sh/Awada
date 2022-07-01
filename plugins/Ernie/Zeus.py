import requests
import json


class Zeus:
    """
    ai.baidu.com/https://ai.baidu.com/ai-doc/wenxin/Il3cbftp9
    Ernie-ViLG
    """
    def __init__(self, access_token):
        self.url = "https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernie/3.0/zeus"
        self.access_token = access_token

    def get_response(self, text):
        payload = {
            'text': text,
            'seq_len': 256,
            'task_prompt': '',
            'dataset_prompt': '',
            'access_token': self.access_token,
            'topk': 10,
            'stop_token': '”',
            'is_unidirectional': 1
        }

        response = requests.request("POST", self.url, data=payload)
        response_text = json.loads(response.text)
        if response_text['code'] == 4001:
            print("请求参数格式错误，不是标准的JSON格式")
        elif response_text['code'] == 4002:
            print("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
        elif response_text['code'] == 4003:
            print("text长度超过模型要求的最大值")
        elif response_text['code'] == 4004:
            print("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
        else:
            return response_text['data']['result']

        return None

if __name__ == "__main__":
    print("====Ernie Zeus Test====")
    print("输入access_token")
    access_token = input("access_token：")
    zeus = Zeus(access_token)

    while True:
        print("输入Q退出")
        prompt = input("256个字以内文本：")
        if prompt.lower() == "q":
            break
        reply = zeus.get_response(prompt)
        if reply is not None:
            print("回复：" + reply)
        