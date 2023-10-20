import io
import logging
import uuid

import pandas as pd
import streamlit as st

from pandasai import SmartDataframe, Agent, Config
from pandasai.callbacks import StdoutCallback
from pandasai.helpers import Logger

from middleware.base import CustomChartsMiddleware
from parser.response_parser import CustomResponseParser
from util import get_open_ai_model, get_ollama_model, get_baidu_as_model, get_prompt_template, get_baidu_qianfan_model

logger = Logger()


class AgentWrapper:
    id: str
    agent: Agent

    def __init__(self) -> None:
        self.agent = None
        self.id = str(uuid.uuid4())

    def get_llm(self):
        op = st.session_state.last_option
        llm = None
        if op == "Ollama":
            llm = get_ollama_model(st.session_state.ollama_model, st.session_state.ollama_base_url)
        elif op == "OpenAI":
            if st.session_state.api_token != "":
                llm = get_open_ai_model(st.session_state.api_token)
        elif op == "Baidu/AIStudio-Ernie-Bot":
            if st.session_state.access_token != "":
                llm = get_baidu_as_model(st.session_state.access_token)
        elif op == "Baidu/Qianfan-Ernie-Bot":
            if st.session_state.client_id != "" and st.session_state.client_secret != "":
                llm = get_baidu_qianfan_model(st.session_state.client_id, st.session_state.client_secret)
        if llm is None:
            st.toast("LLM initialization failed, check LLM configuration", icon="🫤")
        return llm

    def set_file_data(self, file_obj):
        df = pd.read_excel(file_obj)
        grid.dataframe(df)
        counter.info("Total: **%s** Records" % len(df))
        llm = self.get_llm()
        if llm is not None:
            print("llm.type", llm.type)
            config = Config(
                llm=llm,
                callback=StdoutCallback(),
                # middlewares=[CustomChartsMiddleware()],
                response_parser=CustomResponseParser,
                custom_prompts={
                    "generate_python_code": get_prompt_template()
                },
                enable_cache=False,
                verbose=True
            )
            self.agent = Agent(df, config=config, memory_size=1)
            self.agent._lake.add_middlewares(CustomChartsMiddleware())

    def chat(self, prompt):
        if self.agent is None:
            st.toast("LLM initialization failed, check LLM configuration", icon="🫣")
            st.stop()
        else:
            return self.agent.chat(prompt)

    def start_new_conversation(self):
        self.agent.start_new_conversation()
        st.session_state.chat_history = []


@st.cache_resource
def get_agent(agent_id) -> AgentWrapper:
    agent = AgentWrapper()
    st.session_state.llm_ready = True
    return agent


if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False

# if "agent_id" not in st.session_state:
#     st.session_state.agent_id = str(uuid.uuid4())

# page settings
st.set_page_config(page_title="Excel Chat", layout="wide")

# DataGrid
with st.expander("DataGrid Content") as ep:
    grid = st.dataframe(pd.DataFrame(), use_container_width=True)
counter = st.markdown("")

# Sidebar layout
with st.sidebar:
    option = st.selectbox("Choose LLM", ["OpenAI", "Baidu/AIStudio-Ernie-Bot", "Baidu/Qianfan-Ernie-Bot", "Ollama"])

    # Initialize session keys
    if "api_token" not in st.session_state:
        st.session_state.api_token = ""
    if "access_token" not in st.session_state:
        st.session_state.access_token = ""
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = ""
    if "client_id" not in st.session_state:
        st.session_state.client_id = ""
    if "client_secret" not in st.session_state:
        st.session_state.client_secret = ""

    # Initialize model configration panel
    if option == "OpenAI":
        api_token = st.text_input("API Token", st.session_state.api_token)
    elif option == "Baidu/AIStudio-Ernie-Bot":
        access_token = st.text_input("Access Token", st.session_state.access_token)
    elif option == "Baidu/Qianfan-Ernie-Bot":
        client_id = st.text_input("Client ID", st.session_state.client_id)
        client_secret = st.text_input("Client Secret", st.session_state.client_secret)
    elif option == "Ollama":
        ollama_model = st.selectbox(
            "Choose Ollama Model",
            ["starcoder:7b", "codellama:7b-instruct-q8_0", "zephyr:7b-alpha-q8_0"]
        )
        ollama_base_url = st.text_input("Ollama BaseURL", st.session_state.ollama_base_url or "http://localhost:11434")

    # Validation
    info = st.markdown("")
    if option == "OpenAI":
        if not api_token:
            info.error("Invalid API Token")
        if api_token != st.session_state.api_token:
            st.session_state.api_token = api_token
            st.session_state.llm_ready = False
    elif option == "Baidu/AIStudio-Ernie-Bot":
        if not access_token:
            info.error("Invalid Access Token")
        if access_token != st.session_state.access_token:
            st.session_state.access_token = access_token
            st.session_state.llm_ready = False
    elif option == "Baidu/Qianfan-Ernie-Bot":
        if client_id != st.session_state.client_id:
            st.session_state.client_id = client_id
            st.session_state.llm_ready = False
        if client_secret != st.session_state.client_secret:
            st.session_state.client_secret = client_secret
            st.session_state.llm_ready = False
    elif option == "Ollama":
        if ollama_model != st.session_state.ollama_model:
            st.session_state.ollama_model = ollama_model
            st.session_state.llm_ready = False
        if ollama_base_url != st.session_state.ollama_base_url:
            st.session_state.ollama_base_url = ollama_base_url
            st.session_state.llm_ready = False

    if "last_option" not in st.session_state:
        st.session_state.last_option = None

    if option != st.session_state.last_option:
        st.session_state.last_option = option
        st.session_state.llm_ready = False

logger.log(f"st.session_state.llm_ready={st.session_state.llm_ready}", level=logging.INFO)

if not st.session_state.llm_ready:
    st.session_state.agent_id = str(uuid.uuid4())

with st.sidebar:
    st.divider()
    file = st.file_uploader("Upload File", type=["xlsx"])
    if file is None:
        st.session_state.uploaded = False
        if st.session_state.llm_ready:
            get_agent(st.session_state.agent_id).start_new_conversation()

    if file is not None:
        file_obj = io.BytesIO(file.getvalue())
        df = pd.read_excel(file_obj)
        grid.dataframe(df)
        counter.info("Total: **%s** records" % len(df))

        # if not st.session_state.llm_ready:
        st.session_state.agent_id = str(uuid.uuid4())
        get_agent(st.session_state.agent_id).set_file_data(file_obj)
        st.session_state.llm_ready = True

# ChatBox layout
chat_history_key = "chat_history"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if "type" in item and item["type"] == "plot":
            tmp = st.image(item['content'])
        elif "type" in item and item["type"] == "dataframe":
            tmp = st.dataframe(item['content'])
        else:
            st.markdown(item["content"])

prompt = st.chat_input("Input the question here")
if prompt is not None:
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if not st.session_state.llm_ready:
            response = "Please upload the file and configure the LLM well first"
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            tmp = st.markdown(f"Analyzing, hold on pls...")

            response = get_agent(st.session_state.agent_id).chat(prompt)

            if isinstance(response, SmartDataframe):
                tmp.dataframe(response.dataframe)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response.dataframe, "type": "dataframe"})
            elif "type" in response and response["type"] == "plot":
                tmp.image(f"{response['value']}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["value"], "type": "plot"})
            else:
                tmp.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
