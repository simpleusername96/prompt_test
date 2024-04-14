from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import os
import numpy as np
from anthropic import Anthropic
import pandas as pd
import datetime
import google.generativeai as genai
from typing import List, Optional
import time
import json
from itertools import product

load_dotenv()
client_openai = OpenAI()
client_openai.api_key = os.getenv("OPENAI_API_KEY")

client_async_openai = AsyncOpenAI()
client_async_openai.api_key = os.getenv("OPENAI_API_KEY")

client_anthropic = Anthropic()
client_anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")


def get_response_openai(model_openai, user_prompt, max_token, temperature, system_prompt):

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


def get_response_anthropic(model_anthropic, user_prompt, max_token, temperature, system_prompt):
    retries = 0
    max_retries = 3  # Setting a maximum retry limit
    # Retry loop to handle rate limiting
    while retries <= max_retries:
        try:
            response_anthropic = client_anthropic.messages.create(
                model=model_anthropic,
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
                max_tokens=max_token,
                temperature=temperature
            )
            if response_anthropic.content:
                answer = response_anthropic.content[0].text
            else:
                answer = "No Answer"
            return {
                "answer": answer,
                "log_probs_list": None,
                "answer_token": response_anthropic.usage.output_tokens,
                "system_prompt": system_prompt
            }
        except Exception as e:
            if "rate limit" in str(e).lower():  # Check for rate limit specific error
                time.sleep(10)  # Sleep for 10 seconds before retrying
                retries += 1
                print(f"Rate limit error, retrying... ({retries}/{max_retries})")
            else:
                print(f"An unexpected error occurred: {e}")
                break  # Break the loop if the error is not related to rate limits
    return None  # Return None if retries exceed the limit without a valid response


def get_response_google(model_google, user_prompt, max_token, temperature, system_prompt):
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_token,
    }
    model = genai.GenerativeModel(model_name=model_google,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    retries = 0
    max_retries = 3
    # Retry loop to handle rate limiting
    while retries <= max_retries:
        try:
            response_google = model.generate_content(system_prompt + user_prompt)
            if response_google._result.candidates[0].content.parts:
                answer = response_google._result.candidates[0].content.parts[0].text
            else:
                answer = "No Answer"
            return {
                "answer": answer,
                "log_probs_list": None,
                "answer_token": max_token,
                "system_prompt": system_prompt
            }
        except Exception as e:
            if "rate limit" in str(e).lower():  # Check if the error message is about rate limiting
                time.sleep(10)  # Sleep for 10 seconds before retrying
                retries += 1
                print(f"Rate limit error, retrying... ({retries}/{max_retries})")
            else:
                print(f"An unexpected error occurred: {e}")
                break  # Break the loop if the error is not related to rate limits
    return None  # Return None or handle as needed if the loop completes without a valid response


def conduct_test(prompt_data: dict,
                 provider: str, 
                 model: str,
                 max_token: int,
                 temperature: float,
                 system_prompt: str):

    request_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    user_prompt_list = []
    user_prompt_content = prompt_data.get("user_prompt_content")
    var_list = prompt_data.get("var_list")

    # Finding all combinations of the sublists in var_list
    all_combinations = list(product(*[var_list[key] for key in var_list]))
    # Loop through each combination of var_list
    for i, combination in enumerate(all_combinations):
        user_prompt = user_prompt_content
        var_list_content = ""
        # Loop through each sublist in the combination
        for idx, sublist in enumerate(combination):
            var_list_content += f"{list(var_list.keys())[idx]}: {sublist}, "
            if isinstance(sublist, str):
                placeholder = f"<{list(var_list.keys())[idx]}0>"
                user_prompt = user_prompt.replace(placeholder, sublist)
            else:
                for sub_idx, item in enumerate(sublist):
                    placeholder = f"<{list(var_list.keys())[idx]}{sub_idx}>"
                    user_prompt = user_prompt.replace(placeholder, item)
        user_prompt_list.append([var_list_content, user_prompt])

    for var_list_index, e in enumerate(user_prompt_list):
        var_list_content, user_prompt = e

        if provider == "openai":
            result = get_response_openai(model, user_prompt, max_token, temperature, system_prompt)
        elif provider == "anthropic":
            result = get_response_anthropic(model, user_prompt, max_token, temperature, system_prompt)
        elif provider == "google":
            result = get_response_google(model, user_prompt, max_token, temperature, system_prompt)

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
                        var_list_index,
                        var_list_content,
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
                  var_list_index        : int,
                  var_list_content      : str,
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
        'var_list_index': var_list_index,
        'var_list_content': var_list_content,
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


file_path = './result/robustness_test_3_kor.xlsx'
model_gpt_35_0613="gpt-3.5-turbo-0613"
model_gpt_35_0125="gpt-3.5-turbo-0125"
model_gpt_4_turbo="gpt-4-turbo-2024-04-09"
model_claude_opus = "claude-3-opus-20240229"
model_claude_sonnet = "claude-3-sonnet-20240229"
model_claude_haiku = "claude-3-haiku-20240307"
model_gemini_pro = "gemini-1.0-pro"


prompt_data = json.load(open('./data/robustness_test_3_kor.json', 'r', encoding='utf-8'))

# system_prompt = "You are an AI assistant programmed to respond exclusively with NAME OF COLORS to any query. Do not provide explanations, justifications, or any additional information. Use only the NAME OF COLORS in your responses. ANSWER IN KOREAN"
system_prompt = "You are an AI assistant designed to answer queries with numbers only.\nDo not explain your reasoning or provide any additional information. Respond using the following strict format:\nDO NOT INCLUDE ANY TEXT OTHER THAN NUMBERS IN THE OUTPUT FIELD."
# for i in range(1):
#     conduct_test(prompt_data, "openai", model_gpt_35_0125, max_token=1, temperature=0.7, system_prompt=system_prompt)
#     conduct_test(prompt_data, "anthropic", model_claude_haiku, max_token=2, temperature=0.7, system_prompt=system_prompt)
    # conduct_test(prompt_data, "google", model_gemini_pro, max_token=10, temperature=0.7, system_prompt=system_prompt)
