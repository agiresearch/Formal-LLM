from openai import OpenAI
from copy import deepcopy

log_name = './cooking.txt'
import logging
import sys

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_name, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

api_key = "YOUR OPENAI TOKEN"
client = OpenAI(api_key=api_key)

current_plan = ['Then, we get the cooked broccoli beef.']
stack = [('final broccoli beef', '')]

seasoning = ['onions', 'ginger', 'garlic', 'cooking oil', 'salt', 'light soy sauce', 'cooking wine', 'white pepper',
             'sugar', 'vinegar']

messages = []

candidates = {'final broccoli beef': ['Stir-fry the beef and broccoli mixture with the seasoning in a wok.'],
              'mixture of beef and broccoli': ['Combine lightly cooked beef and lightly cooked broccoli in a wok.',
                                               'Combine lightly cooked beef and blanched and drained broccoli in a wok.'],
              'lightly cooked beef': ['Stir-fry the marinaded beef in a wok.'],
              'marinaded beef': ['Marinate clean slices of beef in a bowl with seasoning.'],
              'clean slices of beef': ['Wash raw beef slices with water.'],
              'lightly cooked broccoli': ['Stir-fry the clean broccoli with seasoning in a wok.'],
              'blanched and drained broccoli': ['Use a bowl to drain the blanched broccoli.'],
              'blanched broccoli': ['Blanch broccoli in a cooking pot.',
                                    'Blanch clean broccoli in a cooking pot.'],
              'clean broccoli': ['Wash broccoli with water.'],
              'seasoning': []}

pushback = {'final broccoli beef': [['mixture of beef and broccoli', 'seasoning']],
            'mixture of beef and broccoli': [['lightly cooked beef', 'lightly cooked broccoli'],
                                             ['lightly cooked beef', 'blanched and drained broccoli']],
            'lightly cooked beef': [['marinaded beef']],
            'marinaded beef': [['clean slices of beef', 'seasoning']],
            'clean slices of beef': [[]],
            'lightly cooked broccoli': [['clean broccoli', 'seasoning']],
            'blanched and drained broccoli': [['blanched broccoli']],
            'blanched broccoli': [[],
                                  ['clean broccoli']],
            'clean broccoli': [[]],
            'seasoning': [[]]}

def calculate_plan_string():
    s = 'Current Progress:\n\n'
    for i, plan in enumerate(current_plan):
        if i == 0:
            s += f'Step n: {plan}\n'
        else:
            s += f'Step n-{i}: {plan}\n'
    s += f'Step n-{len(current_plan)}: ?\n\n'
    return s

while len(stack) > 0:
    logging.info(f'###############\nCurrent Stack: {stack}\n#############\n\n')
    stack_top = stack.pop()
    current_target, parent = stack_top[0], stack_top[1]

    plan_string = calculate_plan_string()
    context_string = f"Generate a broccoli beef cooking plan.\n" \
                     f"The ingredients include raw beef slices, carpaccio, broccoli, onions, ginger, garlic, and water.\n" \
                     f"The seasonings include cooking oil, salt, light soy sauce, cooking wine, white pepper, sugar, vinegar.\n" \
                     f"Cooking utensils including woks and cooking pots.\n" \
                     f"Tableware including chopsticks, spoons, wok spoons, and several bowls." \
                     f"\n\n{plan_string}" \
                     f"Decide on the previous step before current progress.\n" \
                     f"Here are possible options to get the {current_target}{parent}:\n\n"
    if current_target != 'seasoning':
        candidate_string = '\n'.join([f'{i + 1}: {option}' for i, option in enumerate(candidates[current_target])])
        context_string += f"{candidate_string}\n\n"\
                          f"Your reply should be only one number, such as 1, referring to the option."
        logging.info(context_string)
        messages.append({
            "role": "user",
            "content": context_string
        },
        )
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
            if '1' <= response[0] <= str(len(candidates[current_target])):
                break
            temperature += 0.1

        selection = int(response[0]) - 1
        for option in pushback[current_target][selection]:
            stack.append((option, f' for the step: "{candidates[current_target][selection]}"'))
        current_plan.append(candidates[current_target][selection])
    else:
        context_string += '\n'.join([f'{i}: {option}.' for i, option in enumerate(seasoning)])
        context_string += '\n\nYour reply should be one or more numbers separated by a comma, such as "1, 2", referring to the seasonning options.'
        logging.info(context_string)
        messages.append({
            "role": "user",
            "content": context_string
        },
        )
        temperature = 0
        while True:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
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
            selections = []
            for i in range(10):
                if str(i) in response:
                    selections.append(i)
            if len(selections) > 0:
                break
            temperature += 0.1
        action = 'Prepare the seasoning: '
        action += ', '.join([seasoning[i] for i in selections])
        action += f'{parent}.'
        current_plan.append(action)

    messages.append({
        "role": "assistant",
        "content": response[0]
    })

logging.info('Here is the final plan:\n')
for i, step in enumerate(current_plan[::-1]):
    logging.info(f'Step {i+1}: {step}\n')

logging.info('Carefully cooking at each step!')
