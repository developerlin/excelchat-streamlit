# ExcelChat
ExcelChat is a AI powered app built on [pandas-ai](https://github.com/gventuri/pandas-ai) and [streamlit](https://github.com/streamlit/streamlit). Upload an excel file, then you can chat with it like chatGPT.

Currently the following models are supported. 
 * OpenAI
 * Ollama: starcoder:7b, codellama:7b-instruct-q8_0, zephyr:7b-alpha-q8_0 
 * Baidu/AIStudio-Ernie-Bot, baidu ernie-bot model for ai studio (single thread mode, not suitable for multi-tenant usage)
 * Baidu/Qianfan-Ernie-Bot, the recommended way to use baidu ernie bot model

Here are some screenshot.

![Screenshot1](docs/images/screen1.png?raw=true)
![Screenshot2](docs/images/screen2.png?raw=true)
![Screenshot3](docs/images/screen3.png?raw=true)

## Demo
 https://excelchat.streamlit.app

## Requirements
Python >= 3.9.

## Quick Install
```shell
pip install -r requirements.txt
```
## Run
Run the following command in the terminal, then you will get the app's link opened in the browser.
```shell
streamlit run Home.py
```
