import json
from transformers import AutoTokenizer, AutoModel
import re
from matplotlib.font_manager import FontProperties

def model_chat(query):
    system_prompt = "你是一个智能的编程助手，你需要使用python完成用户的代码需求。"
    system_info = {
        "role":"system",
        "content" : system_prompt
    }

    i = 0
    while True:
        if i >= 3:
            break

        res,his = model.chat(tokenizer, query, history=[system_info])
        i += 1

        try:
            code = re.findall("```python(.*?)```",res,re.DOTALL)[0]
        except:
            print( "答案中不包含python代码")
            continue
        
        try:
            compile(code,"<string>","exec")
        except:
            print( "生成的python代码有误！")
            continue

        try:
            # if "plt" in code and "matplotlib" in "code":
            #     code += "\nplt.savefig('xxx.png')"
            import numpy as np
            out_dict = {}
            exec(code,{"np":np},out_dict)
            print("执行成功哦～")
            
            result = f"排序前：{out_dict.get('arr1','')}\n排序后：{out_dict.get('arr2','')}"

            return result
        except:
            print( "执行代码报错")
            continue

    return "超时"

    



if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("/mnt/sdb/models/chatglm3_6b_chat", trust_remote_code=True)
    model = AutoModel.from_pretrained("/mnt/sdb/models/chatglm3_6b_chat", trust_remote_code=True).cuda()
    model = model.eval()
    while True:
        query = input("请输入：") # 帮我写一个冒泡排序的代码,排序前的变量为：arr1，排序后的变量为：arr2，不要改变arr1中的值   帮我画一个爱心图,并保存 帮我画一张饼图并保存，【1月 ，2月，3月】，值：【20，30,10】
        response = model_chat(query)
        print(response)