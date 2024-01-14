import time, traceback

import openai
openai.api_key  = "add your openAI api here"


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.3):

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_completion_block_until_succeed(prompt, model="gpt-3.5-turbo", temperature=0.3, sleeptime=3):
    success = False
    while not success:
        try:
            response = get_completion(prompt, model, temperature=temperature)
            success = True
        except:
            traceback.print_exc()
            time.sleep(sleeptime)

    return response
