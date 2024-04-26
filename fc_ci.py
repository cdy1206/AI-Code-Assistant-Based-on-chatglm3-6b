import gradio as gr
import pandas as pd
import akshare as ak
from transformers import AutoTokenizer,AutoModel
import re

def get_image(prompt):
    global model,tokenizer
    system_prompt = "你是一个编程助手，你需要根据用户的需求，生成python代码，并使用matplotlib完成画图"
    system_info = {
        "role":"system",
        "content":system_prompt
    }

    for i in range(3):
        res,his = model.chat(tokenizer,prompt,history=[system_info])

        try:
            if isinstance(res,dict):
                code = re.findall("```python(.*?)```",res["content"],re.DOTALL)[0]
            else:
                code = re.findall("```python(.*?)```",res,re.DOTALL)[0]
        except:
            print("模型不包含python代码")

        try:
            compile(code,"<string>","exec")
        except:
            print("python代码语法错误！")
        
        try:
            code =  """import matplotlib.pyplot as plt\nfrom matplotlib.font_manager import FontProperties\nplt.rcParams['font.sans-serif'] = ['simhei']\nfig = plt.figure()""" + code
            
            out_dict = {}
            code = re.sub(r"plt\.figure\([^)]*\)",r"fig = \g<0>",code)
            exec(code,{},out_dict)
            print("执行成功！")
            return out_dict.get("fig")
        except:
            print("python代码有bug！")

    return None

# 空气质量排行
def air_quality_rank():
    data = ak.air_quality_rank()
    return None,data

# 查询胡润排行榜
def hurun_rank():
    data_ = ak.hurun_rank()
    data = data_.iloc[:7]
    data = data.loc[:,["姓名","财富"]]
    prompt = f"请根据以下数据生成柱状图： \n{data.to_string(index=False)}"
    image = get_image(prompt)
    return image,data

# 查询电影实时票房
def movie_boxoffice_realtime():

    data_ = ak.movie_boxoffice_realtime()
    data = data_.iloc[:7]
    data = data.loc[:,["影片名称","累计票房"]]
    prompt = f"请根据以下数据生成饼图： \n{data.to_string(index=False)}"
    image = get_image(prompt)
    return image,data_

def get_tools():
    tools = [
        {
            "name":"air_quality_rank",
            "description":"空气质量排行",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        },
        {
            "name":"hurun_rank",
            "description":"查询胡润排行榜",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        },
        {
            "name":"movie_boxoffice_realtime",
            "description":"查询电影实时票房",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        }
    ]
    return tools

def model_chat(query):
    global model,tokenizer
    system_prompt = "你是一个人工智能助手，需要根据提供的工具，回复用户的需求，工具如下："
    system_info = {
        "role":"system",
        "content": system_prompt,
        "tools":get_tools()
    }

    res,his = model.chat(tokenizer,query,history=[system_info])

    if isinstance(res,dict):
        func = res.get("name")
        param = res.get("parameters")

        image,data = eval(f"""{func}(**param)""")

    return  image,data

if __name__ == "__main__":

    model_path = "/mnt/sdb/models/chatglm3_6b_chat/"
    model = AutoModel.from_pretrained(model_path,trust_remote_code=True,device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    with gr.Blocks(title="Buding Demo") as demo:
        gr.HTML("<h1 align='center'>Buding Demos</h1> ")
        with gr.Row():
            plot_ = gr.Plot()
            data_show = gr.DataFrame()
        with gr.Row():
            query = gr.Textbox(lines=1,placeholder="请输入提问内容",show_label=False)
            
        with gr.Row():
            button1 = gr.Button("生成答案")
            button2 = gr.Button("重新生成")
        
        query.submit(model_chat,[query],[plot_,data_show])
        button1.click(model_chat,[query],[plot_,data_show])
        button2.click(model_chat,[query],[plot_,data_show])
    demo.launch(server_name='0.0.0.0',server_port=10008,share=False)