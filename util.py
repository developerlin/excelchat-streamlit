from langchain.chat_models import ErnieBotChat
from langchain.llms.ollama import Ollama
from pandasai.llm import OpenAI, LangchainLLM
from pandasai.prompts import GeneratePythonCodePrompt

from llm.ais_erniebot import AIStudioErnieBot


def get_open_ai_model(api_key):
    return OpenAI(api_token=api_key)


def get_ollama_model(model_key, base_url):
    llm = Ollama(model=model_key, base_url=base_url, verbose=True)
    return LangchainLLM(langchain_llm=llm)


def get_baidu_as_model(access_token):
    llm_core = AIStudioErnieBot(access_token=access_token, verbose=True)
    return LangchainLLM(llm_core)


def get_baidu_qianfan_model(client_id, client_secret):
    llm_core = ErnieBotChat(
        model_name="ERNIE-Bot",
        temperature=0.1,
        ernie_client_id=client_id,
        ernie_client_secret=client_secret
    )
    return LangchainLLM(llm_core)


def get_prompt_template():
    instruction_template = """
使用提供的 dataframes ('dfs') 分析这个数据，过程中不要调用 dataframe set_index 对数据排序.
1. 准备: 如果有必要对数据做预处理和清洗
2. 执行: 对数据进行数据分析操作 (grouping, filtering, aggregating, etc.)
3. 分析: 进行实际分析（如果用户要求plot chart，请在代码中添加如下两行代码设置字体, 并将结果保存为图像文件temp_chart.png，并且不显示图表）
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False    
    """
    custom_template = GeneratePythonCodePrompt(custom_instructions=instruction_template)
    return custom_template
