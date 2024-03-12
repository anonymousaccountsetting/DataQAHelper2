import pandas as pd
import os
from docx import Document
import re
import openai
import json
import requests

class SettingForLLM():

    def set_chatGPT(self, background,key, chatmodel="gpt-3.5-turbo", url="https://api.openai.com/v1/chat/completions"):
        openai.api_key = key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        #background=set_for_GPT.set_background(Xcol,ycol,modelname,modelinformation)
        background=background
        messages = [{"role": "system", "content": background}, ]
        return (url, background, chatmodel, headers, messages)

    def set_payload(self, message,GPTmodelname="gpt-3.5-turbo", messages=[]):
        messages.append({"role": "user", "content": message}, )
        payload = {
            "model": GPTmodelname,
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        return (payload, messages)

    def send_response_receive_output(self, URL, headers, payload, messages):
        response = requests.post(URL, headers=headers, json=payload, stream=False)
        print(json.loads(response.content))
        output = json.loads(response.content)["choices"][0]['message']['content']
        messages.append({"role": "assistant", "content": output})
        return (output, messages)

    def save_chat_history_to_docx(self,messages):
        doc = Document()

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'user':
                doc.add_paragraph('User: ' + content)
            elif role == 'assistant':
                doc.add_paragraph('Assistant: ' + content)
            else:
                doc.add_paragraph(content)
        filename="data_report.docx"
        doc.save(filename)


