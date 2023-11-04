import io
import logging
import uuid
from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd
import streamlit as st

from pandasai import SmartDataframe, Agent, Config
from pandasai.callbacks import StdoutCallback
from pandasai.helpers import Logger

from middleware.base import CustomChartsMiddleware
from parser.response_parser import CustomResponseParser
from util import get_open_ai_model, get_ollama_model, get_baidu_as_model, get_prompt_template, get_baidu_qianfan_model

logger = Logger()

matplotlib.rc_file("./.matplotlib/.matplotlibrc");

# page settings
st.set_page_config(page_title="Excel Chat", layout="wide")
st.header("What ExcelChat can do?")
st.text("ExcelChat is a lightweight data analysis app powered by LLM, showcasing how LLM can revolutionize the future"
        "of data analysis.")
st.markdown("""List of todos
 - [x] Add memory
 - [x] Support non-latin text in chart
 - [ ] Sub questions support
""")


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

    def set_file_data(self, df):
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
            self.agent = Agent(df, config=config, memory_size=memory_size)
            self.agent._lake.add_middlewares(CustomChartsMiddleware())
            st.session_state.llm_ready = True

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
    return agent

chat_history_key = "chat_history"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []


if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False

# Description
tab1, tab2 = st.tabs(["Workspace", "Screenshots"])
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/images/short1.png")
    with col2:
        st.image("docs/images/short2.png")

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
        api_token = st.text_input("API Token", st.session_state.api_token, type="password", placeholder="Api token")
    elif option == "Baidu/AIStudio-Ernie-Bot":
        access_token = st.text_input("Access Token", st.session_state.access_token, type="password",
                                     placeholder="Access token")
    elif option == "Baidu/Qianfan-Ernie-Bot":
        client_id = st.text_input("Client ID", st.session_state.client_id, placeholder="Client ID")
        client_secret = st.text_input("Client Secret", st.session_state.client_secret, type="password",
                                      placeholder="Client Secret")
    elif option == "Ollama":
        ollama_model = st.selectbox(
            "Choose Ollama Model",
            ["starcoder:7b", "codellama:7b-instruct-q8_0", "zephyr:7b-alpha-q8_0"]
        )
        ollama_base_url = st.text_input("Ollama BaseURL", st.session_state.ollama_base_url,
                                        placeholder="http://localhost:11434")

    memory_size = st.selectbox("Memory Size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=9)

    if st.button("+ New Chat"):
        st.session_state.llm_ready = False
        st.session_state[chat_history_key] = []

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

    if "last_memory_size" not in st.session_state:
        st.session_state.last_memory_size = None

    if memory_size != st.session_state.last_memory_size:
        st.session_state.last_memory_size = memory_size
        st.session_state.llm_ready = False

logger.log(f"st.session_state.llm_ready={st.session_state.llm_ready}", level=logging.INFO)

if not st.session_state.llm_ready:
    st.session_state.agent_id = str(uuid.uuid4())

with st.sidebar:
    st.divider()
    file = st.file_uploader("Upload File", type=["xlsx", "csv"])
    if file is None:
        st.session_state.uploaded = False
        if st.session_state.llm_ready:
            get_agent(st.session_state.agent_id).start_new_conversation()

    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if file is not None:
        file_obj = io.BytesIO(file.getvalue())
        file_ext = Path(file.name).suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
        grid.dataframe(df)
        counter.info("Total: **%s** records" % len(df))

        if file != st.session_state.last_file or st.session_state.llm_ready is False:
            # if not st.session_state.llm_ready:
            st.session_state.agent_id = str(uuid.uuid4())
            get_agent(st.session_state.agent_id).set_file_data(df)

        st.session_state.last_file = file

with st.sidebar:
    st.markdown("""
    <style>
        .tw_share {
            # position: fixed;
            display: inline-block;
            # left: 240px;
            # bottom: 20px;
            cursor: pointer;
        }
        
        .tw_share a {
            text-decoration: none;
        }
        
        .tw_share span {
            color: white;
        }
        
        .tw_share span {
            margin-left: 2px;
        }
        
        .tw_share:hover svg path {
            fill: #1da1f2;
        }
        
        .tw_share:hover span {
            color: #1da1f2;
        }
    </style>
    <div class="tw_share">
        <a target="_blank" href="https://twitter.com/intent/tweet?url=https://excelchat.streamlit.app/&text=ExcelChat%20-%20LLM%20revolutionize%20the%20future%20of%20data%20analysis"><svg width="24" height="18" viewBox="0 0 24 18" fill="none" xmlns="http://www.w3.org/2000/svg" class="socialCallout_Icon__cnbtq"><path d="M22.9683 0.526535C22.5928 1.56402 21.8955 2.36683 20.9031 2.95968C21.368 2.91851 21.8239 2.83205 22.271 2.71678C22.7225 2.60562 23.1561 2.45741 23.5986 2.28449C23.5986 2.32566 23.5763 2.34625 23.5629 2.36683C22.946 3.19435 22.1994 3.91071 21.3233 4.51179C21.2428 4.56943 21.207 4.61883 21.2115 4.71764C21.2651 6.1133 21.0595 7.47603 20.6214 8.81405C20.1029 10.3867 19.3072 11.8277 18.2119 13.1287C16.9334 14.6478 15.3554 15.8376 13.469 16.6693C12.3156 17.1798 11.1042 17.5174 9.8391 17.7068C9.10597 17.8138 8.37284 17.8673 7.63078 17.8797C5.98571 17.9044 4.38982 17.6656 2.83863 17.1674C1.87305 16.8545 0.952171 16.4428 0.0804651 15.9365C0.0536434 15.92 0.0178811 15.9118 0 15.8788C2.633 16.1217 5.00672 15.5206 7.11669 14.0426C5.87395 13.9809 4.79662 13.5774 3.89362 12.787C3.29907 12.2682 2.87886 11.6465 2.62406 10.9178C3.34824 11.0372 4.05902 11.0208 4.79215 10.8478C3.3393 10.5308 2.24408 9.7939 1.52437 8.61232C1.10863 7.93301 0.920879 7.19607 0.925349 6.40149C1.60483 6.7432 2.32008 6.92847 3.10238 6.9614C2.93698 6.84201 2.78052 6.7432 2.64194 6.62793C1.82835 5.96921 1.29638 5.15404 1.07287 4.18243C0.822532 3.09554 0.992403 2.06218 1.5646 1.08645C1.61824 0.991756 1.62271 0.995873 1.69424 1.0741C2.70452 2.19392 3.88468 3.13259 5.23917 3.88189C6.67413 4.67647 8.22085 5.20756 9.87039 5.4834C10.4113 5.57398 10.9567 5.63161 11.5065 5.66043C11.5914 5.66455 11.6183 5.66043 11.6004 5.56574C11.3053 4.10009 11.6674 2.79088 12.7403 1.67106C13.5271 0.843544 14.5284 0.357738 15.7131 0.221878C17.1167 0.0571976 18.3729 0.398908 19.4726 1.23054C19.6022 1.32935 19.7184 1.43227 19.8347 1.54343C19.8749 1.58049 19.9106 1.58872 19.9732 1.58049C21.0059 1.38699 21.9714 1.04116 22.8834 0.559471C22.9057 0.543003 22.9236 0.522418 22.9683 0.526535Z" fill="#FFFFFF"></path></svg><span>Share & Talk w/ me</span></a>
    </div>
    """, unsafe_allow_html=True)

# ChatBox layout

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
            elif isinstance(response, Dict) and "type" in response and response["type"] == "plot":
                tmp.image(f"{response['value']}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["value"], "type": "plot"})
            else:
                tmp.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
