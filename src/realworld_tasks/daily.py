from openai import OpenAI
from copy import deepcopy

log_name = './daily.txt'
import logging
import sys
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

api_key = "YOUR OPENAI TOKEN"
client = OpenAI(api_key=api_key)
worktime = {
    'Eating breakfast.': 1,
    'Eating lunch.': 1,
    'Eating supper.': 1,
    'Playing basketball.': 2,
    'Grocery shopping.': 1,
    'House cleaning.': 1,
    'Doing homework.': 3,
    'Turning on the washer/laundry machine.': 0,
    'Doing nothing for one hour.': 1,
}

current_plan = []
stack = 'D'
persistent_tools = []
persistent_stack = []
persistent_meal = []
meal_stack = ''

def calculate_plan_string():
    s = 'Current Progress:\n\n'
    hour = 20
    for plan in current_plan:
        start_hour = hour - worktime[plan]
        s += f'{start_hour}:00 - {hour}:00 {plan}\n'
        hour = start_hour
    s += '\n'
    return s

def calculate_possible_tools(time):
    result = ['Backtrace']
    for tool in worktime.keys():
        if tool in current_plan:
            continue
        if tool == 'Eating breakfast.':
            if (stack.endswith('E') or stack.endswith('EH')) and len(meal_stack) == 0:
                result.append(tool)
        elif tool == 'Eating lunch.':
            if (stack.endswith('Z') or stack.endswith('ZH')) and len(meal_stack) == 0:
                result.append(tool)
        elif tool == 'Eating supper.':
            if (stack.endswith('D') or stack.endswith('DH'))  and len(meal_stack) == 0:
                result.append(tool)
        elif tool == 'Playing basketball.':
            if time == 15 and not stack.endswith('H'):
                result.append(tool)
        elif tool == 'Grocery shopping.':
            if not stack.endswith('H'):
                result.append(tool)
        elif tool == 'Turning on the washer/laundry machine.':
            if time != 20:
                result.append(tool)
        else:
            result.append(tool)
    return result

current_hour = 20
messages = []

while True:
    if 'N' in stack and 'E' not in stack and 'Z' not in stack and 'D' not in stack and current_hour >= 10 and \
        ('Doing nothing for one hour.' in current_plan and len(current_plan) == len(worktime.keys())
         or 'Doing nothing for one hour.' not in current_plan and len(current_plan) == len(worktime.keys()) - 1):
        logging.info('###### Find Valid Plan: ######')
        logging.info(current_plan)
        logging.info(calculate_plan_string())
        break
    if current_hour <= 10:
        # Backtrace
        logging.info('##### Backtrace #####')
        activity = current_plan.pop()
        current_hour += worktime[activity]
        possible_tools = persistent_tools.pop()
        stack = persistent_stack.pop()
        meal_stack = persistent_meal.pop()
        messages.pop()
        messages.pop()
    else:
        possible_tools = calculate_possible_tools(current_hour)

    while len(possible_tools) <= 1:
        # Backtrace
        logging.info('##### Backtrace #####')
        if current_hour == 20:
            logging.info('No more states can be backtrace. Quit!')
            exit(0)
        activity = current_plan.pop()
        current_hour += worktime[activity]
        possible_tools = persistent_tools.pop()
        stack = persistent_stack.pop()
        meal_stack = persistent_meal.pop()
        messages.pop()
        messages.pop()

    tool_cnt = len(possible_tools)
    if len(current_plan) == 0:
        plan_string = ''
    else:
        plan_string = calculate_plan_string()

    candidate_string = ''
    for i, candidate in enumerate(possible_tools[1:]):
        candidate_string += f'{i + 1}. {candidate}\n'

    content_string = f"Generate a plan for activities between 10:00 and 20:00. \n\nBreakfast, lunch, and supper need 1 hour each.\n\n"\
                     f"Outdoor activities: basketball playing 13:00 - 15:00; do grocery shopping needs 1 hour.\n\n"\
                     f"Indoor activities: house cleaning needs 1 hour; homework needs three hours; turning on the washer/laundry machine needs 0 minutes but needs to stay home for one hour.\n\n"\
                     f"Other constrain:\n"\
                     f"Cannot play basketball within an hour after a meal.\n" \
                     f"\n{plan_string}"\
                     f"Let's start planning from the end. Decide on the activity ending at {current_hour}:00.\n\n"\
                     f"Here are possible options:\n\n"\
                     f"{candidate_string}\n"\
                     f"Your reply should be only one number, such as 1, referring to the option. "
    logging.info(f'###############\nCurrent Stack: {stack}\n\nMeal Stack: {meal_stack}\n#############\n\n')
    logging.info(content_string)
    messages.append({
        "role": "user",
        "content": content_string
    },
    )
    temperature = 0
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["."]
        )

        response = response.choices[0].message.content
        logging.info(response)
        if '1' <= response[0] < str(tool_cnt):
            break
        temperature += 0.1

    selection = int(response[0])
    selection = possible_tools.pop(selection)
    current_plan.append(selection)
    persistent_tools.append(deepcopy(possible_tools))
    persistent_stack.append(deepcopy(stack))
    persistent_meal.append(deepcopy(meal_stack))
    current_hour -= worktime[selection]
    messages.append({
        "role": "assistant",
        "content": response[0]
    })

    if selection == 'Eating breakfast.':
        stack = stack.replace('E', 'N')
        stack = stack.replace('H', '')
        meal_stack += 'F' * 3
    elif selection == 'Eating lunch.':
        stack = stack.replace('Z', 'E')
        stack = stack.replace('H', '')
        meal_stack += 'F' * 3
    elif selection == 'Eating supper.':
        stack = stack.replace('D', 'Z')
        stack = stack.replace('H', '')
        meal_stack += 'F' * 3
    elif selection == 'Turning on the washer/laundry machine.':
        stack += 'H'
    else:
        meal_stack = meal_stack[:-worktime[selection]]
        if selection == 'Playing basketball.':
            stack += 'F'
        elif selection in ['House cleaning.', 'Doing homework.', 'Doing nothing for one hour.']:
            stack = stack.replace('H', '')
            stack = stack.replace('F', '')
        elif selection == 'Grocery shopping.':
            stack = stack.replace('F', '')
