import json
from transformers import AutoTokenizer, AutoModel
import re
from matplotlib.font_manager import FontProperties

def model_chat(query):
    system_prompt = "Your are a best programmer assistant,your task is to meet customers requirement!"
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
            print( "The answer does not contain any python code!!!")
            continue
        
        try:
            compile(code,"<string>","exec")
        except:
            print( "Your python code have something wrong!!!")
            continue

        try:
            # if "plt" in code and "matplotlib" in "code":
            #     code += "\nplt.savefig('xxx.png')"
            import numpy as np
            out_dict = {}
            exec(code,{"np":np},out_dict)
            print("Success exc!!!")
            
            result = f"Before sorted：{out_dict.get('arr1','')}\n After sorted{out_dict.get('arr2','')}"

            return result
        except:
            print( "Bad exc for wrong code!!!")
            continue

    return "There happened the overtime!!!"

    



if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("/mnt/sdb/models/chatglm3_6b_chat", trust_remote_code=True)
    model = AutoModel.from_pretrained("/mnt/sdb/models/chatglm3_6b_chat", trust_remote_code=True).cuda()
    model = model.eval()
    while True:
        query = input("Please input:")   ##your real requirement
        response = model_chat(query)
        print(response)
