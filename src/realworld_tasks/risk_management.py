from openai import OpenAI
from copy import deepcopy
import logging
import sys

def get_valid_list_response(max_num):
    temperature = 0
    selects = []
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["."]
        )

        response = response.choices[0].message.content
        logging.info(response)
        for i in range(max_num):
            if str(i) in response:
                selects.append(i)
        if len(selects) > 0:
            break
        temperature += 0.1
    return selects

def get_valid_response(max_num):
    temperature = 0
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["."]
        )

        response = response.choices[0].message.content
        logging.info(response)
        if '1' <= response[0] <= str(max_num):
            break
        temperature += 0.1
    return int(response[0])


def first_condition():
    msg = f'Question: Is {company_B} a 100% subsidiary of {seller_A}?\n' \
          f'1: Yes.\n' \
          f'2: No.\n' \
          f'3: Insufficient information to make a judgment.\n\n' \
          f'Your answer should be only one number, such as 1, referring to the option.'
    logging.info(msg)
    messages.append({
        "role": "user",
        "content": msg,
    })

    res = get_valid_response(3)

    messages.append({
        "role": "assistant",
        "content": str(res),
    })
    if res == 1:
        current_plan.append(f'First, based on current information, we believe {seller_A} is entitled to sell {company_B}.')
        return True
    if res == 2:
        current_plan.append(f'Sorry, based on current information, we believe {seller_A} is NOT entitled to sell {company_B}.')
        return False
    if res == 3:
        current_plan.append(f'We need more information to ensure {seller_A} is entitled to sell {company_B}. '
                            f'But we assume we have found out enough information to proceed the risk assessment process.')
    return True

def second_condition():
    msg = f'Question: Does {buyer_C} buying {company_B} trigger mandatory antitrust filing?\n' \
          f'1: Yes.\n' \
          f'2: No.\n' \
          f'Your answer should be only one number, such as 1, referring to the option.'
    logging.info(msg)
    messages.append({
        "role": "user",
        "content": msg,
    })

    res = get_valid_response(2)

    messages.append({
        "role": "assistant",
        "content": str(res),
    })
    if res == 1:
        region = ['Europe', 'the United States', 'China']
        region_string = '\n'.join([f'{i}: {option}.\n' for i, option in enumerate(region)])
        msg = f'Which of the following countries or regions will require mandatory antitrust filing?' \
              f'\n{region_string}\n\n' \
              f'Your reply should be one or more numbers separated by a comma, such as "1, 2", referring to the options.'
        messages.append({
            "role": "user",
            "content": msg,
        })
        logging.info(msg)
        res = get_valid_list_response(len(region))
        res_string = ', and '.join([region[i] for i in res])
        current_plan.append(f'Then, we need to submit the regional antitrust filing of {res_string}. '
                            f'But we assume the filings will be approved to proceed the risk assessment process.')
    if res == 2:
        current_plan.append(f'Based on current information, we believe no mandatory antitrust filing is needed.')
    return True

def third_condition():
    msg = f'Question: Does {company_B} have material outstanding liabilities or if {company_B} is not in good standing?\n' \
          f'1: Yes.\n' \
          f'2: No.\n' \
          f'3: Insufficient information to make a judgment.\n\n' \
          f'Your answer should be only one number, such as 1, referring to the option.'

    messages.append({
        "role": "user",
        "content": msg,
    })
    logging.info(msg)
    res = get_valid_response(3)

    messages.append({
        "role": "assistant",
        "content": str(res),
    })
    if res == 1:
        current_plan.append(f"Based on current information, the deal is dead because {company_B}'s operation status "
                            f'is bad.')
        return False
    if res == 2:
        current_plan.append(f'Then, based on current information, we believe {company_B} is in good standing without '
                            f'material outstanding liabilities.')
        return True
    if res == 3:
        current_plan.append(f'We need more information to ensure {company_B} is in good standing without material outstanding liabilities. '
                            f'But we assume we have found out enough information to prove it to proceed the risk assessment process.')
    return True

def fourth_condition():
    msg = f'Question: Is {company_B} or {buyer_C} in sensitive industries or related to national security issues ' \
          f'that may trigger CFIUS review in the US or the foreign investment review in China?\n' \
          f'1: Yes.\n' \
          f'2: No.\n' \
          f'Your answer should be only one number, such as 1, referring to the option.'
    logging.info(msg)
    messages.append({
        "role": "user",
        "content": msg,
    })

    res = get_valid_response(2)

    messages.append({
        "role": "assistant",
        "content": str(res),
    })
    if res == 1:
        current_plan.append(f'Then, based on current information, we need to submit the CFIUS filings in the US and '
                            f'the foreign investment filings in China. '
                            f'But we assume both the CFIUS and Chinese government will clear the transaction '
                            f'to proceed the risk assessment process.')
    if res == 2:
        current_plan.append(f'Then, based on current information, we believe neither CFIUS filings nor foreign '
                            f'investment filings in China are needed.')
    return True

log_name = './risk.txt'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_name, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

api_key = "YOUR OPENAI TOKEN"
client = OpenAI(api_key=api_key)

current_plan = []
messages = []

seller_A = 'Shareholders of Blizzard Entertainment'
company_B = 'Blizzard Entertainment'
buyer_C = 'Microsoft'

system_msg = f'You are a legal counsel of {buyer_C}. You are negotiating a transaction with {seller_A}.' \
             f'{seller_A} proposes to sell {company_B}.'

messages.append({
    "role": "system",
    "content": system_msg,
})

logging.info(system_msg)

task_msg = f'You are designing legal due diligence process to assess the risk of buying {company_B}. ' \
           f'I will prepare the plan for you. You only need to answer my questions.'

messages.append({
    "role": "user",
    "content": task_msg,
})

result = first_condition()

if result:
    conditions = ['Anti-trust filing', 'Outstanding liabilities or operation status',
                  'Sensitive industry or national security issues']
    while len(conditions) > 0:
        selections = '\n'.join([f'{i + 1}: {option}.' for i, option in enumerate(conditions)])
        selection_msg = 'Based on current information, select the option that may pose the greatest threat to this transaction from ' \
                        f'the following options.\n{selections}\n\nYour answer should be only one number, such as 1, ' \
                        f'referring to the option.'
        messages.append({
            "role": "user",
            "content": selection_msg,
        })
        logging.info(selection_msg)
        res = get_valid_response(len(conditions))
        selection = conditions[res - 1]
        logging.info(selection)
        conditions.remove(selection)
        logging.info(conditions)
        if selection == 'Anti-trust filing':
            result = second_condition()
        elif selection == 'Outstanding liabilities or operation status':
            result = third_condition()
        elif selection == 'Sensitive industry or national security issues':
            result = fourth_condition()
        if not result:
            break

    if len(conditions) == 0:
        current_plan.append("Pass the preliminary risk analysis. Details to be discussed in the transaction documents.")

logging.info(f'Here is the final plan for {buyer_C}:\n')
for i, step in enumerate(current_plan):
    logging.info(f'{i+1}: {step}\n')

logging.info('Note: This is a risk assessment process provided based on current information. '
             'Please ensure the accuracy of the provided information and possible additional supplemental information.')
