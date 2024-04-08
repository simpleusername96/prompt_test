from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import datetime
from typing import List, Text, Optional
import time
import json

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

load_dotenv()
client_openai = OpenAI()
client_openai.api_key = os.getenv("OPENAI_API_KEY")

client_anthropic = Anthropic()
client_anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

request_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

file_path = f'./result/robustness_test_1_kor_{request_date}.xlsx'
model_gpt_35_0613="gpt-3.5-turbo-0613"
model_gpt_35_0125="gpt-3.5-turbo-0125"
model_claude_sonnet = "claude-3-sonnet-20240229"
model_claude_haiku = "claude-3-haiku-20240307"
model_gemini_pro = "gemini-1.0-pro"


user_prompt_content_list_1_kor, user_prompt_style_list_1_kor = json.load(open('./data/robustness_test_1_kor.json', 'r')).values()
# user_prompt_content_list_1_eng, user_prompt_style_list_1_eng = json.load(open('./data/robustness_test_1_eng.json', 'r')).values()
# user_prompt_content_list_1_2, user_prompt_style_list_1_2 = json.load(open('./data/robustness_test_1_2.json', 'r')).values()

def paraphrase_question(seed_content: str, 
                        seed_style: str, 
                        variation_num: int,
                        file_path: str,
                        max_tokens_content: int = 150,
                        temperature_content: float = 0.7,
                        max_tokens_style: int = 10,
                        temperature_style: float = 0.7):
    """
    Generate paraphrased versions of a given question and its template using GPT-4.

    Parameters:
    seed_content (str): The original content of the question.
    seed_style (str): The original style/format of the question.
    variation_num (int): The number of variations to generate.
    file_path (str): The file path where the result will be saved.

    The function creates two lists of paraphrases:
    1. Paraphrased questions based on the seed_content.
    2. Paraphrased question templates based on the seed_style.
    The results are then saved in a JSON file at file_path.
    """

    # List to store the paraphrased question content.
    previous_prompt_content = [seed_content]

    # Placeholder for system prompt.
    system_prompt = ""

    model_openai_paraphrase = "gpt-4-turbo-preview"

    # Loop to generate paraphrased questions based on content.
    for i in range(variation_num):
        user_prompt_for_content = f"""
            Please create a paraphrased version of the following question. Keep the original meaning intact, and ensure it's phrased differently from the previous versions listed below:
            Original Question: '{seed_content}'
            Previous Versions: {', '.join(previous_prompt_content)}
        """
        response_openai = client_openai.chat.completions.create(
            model=model_openai_paraphrase,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_content}
            ],
            max_tokens=max_tokens_content,
            temperature=temperature_content
        )
        
        # Append the new paraphrased content to the list.
        previous_prompt_content.append(response_openai.choices[0].message.content)

    # List to store the paraphrased question style.
    previous_prompt_style = [seed_style]

    # Loop to generate paraphrased questions based on style.
    for i in range(variation_num):
        user_prompt_for_style = f"""
            Could you reformat the following question template while preserving its original intent? Avoid formats similar to the previous ones listed:
            Original Template: '{seed_style}'
            Previously Generated Templates: {', '.join(previous_prompt_style)}
            Keep the essence but change the style.
        """
        response_openai = client_openai.chat.completions.create(
            model=model_openai_paraphrase,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_style}
            ],
            max_tokens=max_tokens_style,
            temperature=temperature_style
        )
        
        # Append the new style to the list.
        previous_prompt_style.append(response_openai.choices[0].message.content)

    # Creating a result dictionary to store both content and style lists.
    result = {
        "user_prompt_content_list": previous_prompt_content,
        "user_prompt_style_list": previous_prompt_style
    }

    # Save the results to a file in JSON format.
    with open(file_path, mode='w' if os.path.isfile(file_path) else 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)



def get_response_anthropic(model_anthropic, user_prompt, max_token, temperature):
    system_prompt = """"You are an AI assistant designed to answer queries with numbers only. 
    Do not explain your reasoning or provide any additional information. Respond using the following strict format:
    DO NOT INCLUDE ANY TEXT OTHER THAN NUMBERS IN THE OUTPUT FIELD."""


    cnt = 0
    while True:
        try:
            if cnt > 3:
                return
            response_anthropic = client_anthropic.messages.create(
                model=model_anthropic,  
                system = system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
                max_tokens=max_token,
                temperature=temperature
            )
            break
        except Exception as e:
            time.sleep(5)
            print(e)
            cnt += 1

    result = {
        "answer": response_anthropic.content[0].text,
        "log_probs_list": None,
        "answer_token": response_anthropic.usage.output_tokens,
        "system_prompt": system_prompt
    }
    return result

def get_response_openai(model_openai, user_prompt, max_token, temperature):
    system_prompt = """"You are an AI assistant designed to answer queries with numbers only. 
    Do not explain your reasoning or provide any additional information. Respond using the following strict format:
    DO NOT INCLUDE ANY TEXT OTHER THAN NUMBERS IN THE OUTPUT FIELD."
    """

    response_openai = client_openai.chat.completions.create(
        model=model_openai,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_token,
        temperature=temperature,
        logprobs=True
    )
    result = {
        "answer": response_openai.choices[0].message.content,
        "log_probs_list": [token.logprob for token in response_openai.choices[0].logprobs.content],
        "answer_token": response_openai.usage.completion_tokens,
        "system_prompt": system_prompt
    }

    return result

def get_response_google(model_google, user_prompt, max_token, temperature):
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]

    system_prompt = """"You are an AI assistant designed to answer queries with numbers only. 
    Do not explain your reasoning or provide any additional information. Respond using the following strict format:
    DO NOT INCLUDE ANY TEXT OTHER THAN NUMBERS IN THE OUTPUT FIELD."""

    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_token,
    }
    model = genai.GenerativeModel(model_name = model_google,
                                  generation_config = generation_config,
                                  safety_settings = safety_settings)
    
    cnt = 0
    while True:
        try:
            if cnt > 3:
                return
            response_google = model.generate_content(system_prompt + user_prompt)
            break
        except Exception as e:
            time.sleep(5)
            print(e)
            cnt += 1


    
    if not len(response_google._result.candidates[0].content.parts) :
        answer = "No Answer"
    else:
        answer = response_google._result.candidates[0].content.parts[0].text
    
    result = {
        "answer": answer,
        "log_probs_list": None,
        "answer_token": max_token,
        "system_prompt": system_prompt
    }
    return result

def conduct_test(user_prompt_content_list: List[Text], 
                 user_prompt_style_list: List[Text], 
                 provider: str, 
                 model: str,
                 max_token: int,
                 temperature: float):
    """
    Conduct a test using OpenAI's GPT-3.5 engine.
    """

    for i, user_prompt_content in enumerate(user_prompt_content_list):
        for j, user_prompt_style in enumerate(user_prompt_style_list):
            user_prompt_content_id, user_prompt_style_id = i, j
            user_prompt = user_prompt_style.replace("<text>", user_prompt_content)

            if provider == "openai":
                result = get_response_openai(model, user_prompt, max_token, temperature)
            elif provider == "anthropic":
                result = get_response_anthropic(model, user_prompt, max_token, temperature)
            elif provider == "google":
                result = get_response_google(model, user_prompt, max_token, temperature)

            answer = result["answer"]
            log_probs_list = result["log_probs_list"]
            answer_token = result["answer_token"]
            system_prompt = result["system_prompt"]
            save_as_excel(request_date,
                          provider,
                          model, 
                          temperature, 
                          max_token, 
                          system_prompt, 
                          user_prompt, 
                          user_prompt_content_id, 
                          user_prompt_content, 
                          user_prompt_style_id, 
                          user_prompt_style, 
                          answer, 
                          log_probs_list, 
                          answer_token)

def save_as_excel(request_date          : datetime,
                  provider              : str,
                  model                 : str,
                  temperature           : float, 
                  max_token             : int, 
                  system_prompt         : str, 
                  user_prompt           : int, 
                  user_prompt_content_id: int, 
                  user_prompt_content   : str,
                  user_prompt_style_id  : int, 
                  user_prompt_style     : str, 
                  answer                : str, 
                  log_probs_list        : Optional[List[float]], 
                  answer_token          : int):

    data = {
        'request_date': request_date,
        'provider': provider,
        'model': model,
        'temperature': temperature,
        'max_token': max_token,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'user_prompt_content_id': user_prompt_content_id,
        'user_prompt_content': user_prompt_content,
        'user_prompt_style_id': user_prompt_style_id,
        'user_prompt_style': user_prompt_style,
        'answer': answer,
        'answer_token': answer_token,
        'logprobs_mean': f"{np.mean(log_probs_list):.4f}" if log_probs_list is not None else log_probs_list,
    }
    new_df = pd.DataFrame([data])
    if os.path.isfile(file_path):
        # Append new DataFrame to the existing Excel file
        with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            new_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # Create a new Excel file
        new_df.to_excel(file_path, index=False)


