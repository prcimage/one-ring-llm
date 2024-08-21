import json
import os
import re

import numpy as np
import yaml
from dotenv import load_dotenv
from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError
from openai import OpenAI
from termcolor import colored

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

DEBUG = False
PATIENT = "patient_0"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def input_multi(msg: str):
    print(msg, end="")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)

    return "\n".join(contents)


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        try:
            role = message["role"]
        except:
            role = message.role

        if role == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[role]))
        elif role == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[role]))
        elif role == "assistant" and message.function_call:
            print(colored(f"assistant: {message.function_call}\n", role_to_color[role]))
        elif role == "assistant" and not message.function_call:
            print(colored(f"assistant: {message.content}\n", role_to_color[role]))
        elif role == "function":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    role_to_color[role],
                )
            )


with open("data/system_prompt.md", "r") as file:
    system_prompt = file.read()

with open("data/system_functions.json", "r") as file:
    system_functions = json.load(file)

with open("data/care_pathways.yaml", "r") as file:
    care_pathways = yaml.safe_load(file)

with open("data/med_models.json", "r") as file:
    med_models = json.load(file)

with open("data/diagnostic_tests.json", "r") as file:
    diagnostic_tests = json.load(file)

with open("data/clinics.json", "r") as file:
    clinics = json.load(file)

with open(f"data/{PATIENT}/ehr.json", "r") as file:
    ehr = json.load(file)

for i in range(len(med_models)):
    if med_models[i].get("embedding") is None:
        med_models[i]["embedding"] = get_embedding(
            med_models[i]["name"] + "\n" + med_models[i]["description"]
        )

for i in range(len(diagnostic_tests)):
    if diagnostic_tests[i].get("embedding") is None:
        diagnostic_tests[i]["embedding"] = get_embedding(
            diagnostic_tests[i]["name"] + "\n" + diagnostic_tests[i]["description"]
        )

for i in range(len(care_pathways)):
    if care_pathways[i].get("embedding") is None:
        care_pathways[i]["embedding"] = get_embedding(
            care_pathways[i]["name"] + "\n" + care_pathways[i]["content"]
        )

for i in range(len(ehr)):
    if ehr[i].get("embedding") is None:
        ehr[i]["embedding"] = get_embedding(ehr[i]["name"] + "\n" + ehr[i]["content"])

for i in range(len(clinics)):
    if clinics[i].get("embedding") is None:
        clinics[i]["embedding"] = get_embedding(
            clinics[i]["name"] + "\n" + clinics[i]["description"]
        )


def list_to_obj(obj):
    return {func["name"]: func["parameters"]["properties"] for func in obj}


def merge_function_lists(funclists):
    res = {}

    for funclist in funclists:
        res.update(list_to_obj(funclist))

    return res


functions = merge_function_lists([med_models, diagnostic_tests, system_functions])


def search(arr, query, n=3):
    embedding = get_embedding(query)

    ranked_models = sorted(
        arr, key=lambda x: cosine_similarity(x["embedding"], embedding), reverse=True
    )

    return [
        {k: v for k, v in model.items() if k != "embedding"}
        for model in ranked_models[:n]
    ]


with open(f"data/{PATIENT}/primary_care.md", "r") as file:
    patient_case = file.read()

user_prompt = input("What do you want to ask?\n")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": patient_case},
    {"role": "user", "content": user_prompt},
]


def call_llm(messages, functions, model="gpt-4-1106-preview"):
    return client.chat.completions.create(
        model=model,
        # model="gpt-3.5-turbo-0125",
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.4,
        max_tokens=1024,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        seed=2024,
    )


while True:
    pretty_print_conversation(messages)

    response = call_llm(messages, system_functions)

    if DEBUG:
        print(response)
    else:
        os.system("clear")

    response_message = response.choices[0].message

    messages.append(response_message)

    pretty_print_conversation(messages)

    if not response_message.function_call:
        user_prompt = input("What do you want to ask?\n")
        match = re.search(r"INSERT (\w+)", user_prompt)

        print(match)

        if match:
            file = match.group(1)

            print(file)

            with open(f"data/{PATIENT}/{file}.md", "r") as file:
                text = file.read()

            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "user", "content": user_prompt})
    else:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)

        if functions.get(function_name) is None:
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(
                        {
                            "error": "No such function",
                            "details": "The function requested does not exist. Try using the applicable serach function to find a suitable function.",
                        },
                        indent=4,
                    ),
                }
            )
            continue
        try:
            validate(instance=function_args, schema=functions[function_name])
        except ValidationError as e:
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(
                        {"error": "Validation Failed", "details": e.message}, indent=4
                    ),
                }
            )
            continue
        except SchemaError as e:
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(
                        {"error": "Validation Failed", "details": e.message}, indent=4
                    ),
                }
            )
            continue

        if function_name == "request_diagnostic_test" or function_name == "run_model":
            subfunction_name = function_args["name"]
            subfunction_args = function_args.get("arguments")

            if functions.get(subfunction_name) is None:
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(
                            {
                                "error": "No such function",
                                "details": f"The requested function ({subfunction_name}) to use does not exist",
                            },
                            indent=4,
                        ),
                    }
                )
                continue

            if subfunction_args is None:
                subfunction_args = {}

            if functions[subfunction_name] != {} and subfunction_args == {}:
                messages.append(
                    {
                        "role": "function",
                        "name": subfunction_name,
                        "content": json.dumps(
                            {
                                "error": "Validation Failed",
                                "details": f'Function call requries arguments in the "arguments" key: {json.dumps(functions[subfunction_name])}',
                            },
                            indent=4,
                        ),
                    }
                )
                continue

            try:
                validate(instance=subfunction_args, schema=functions[subfunction_name])
            except ValidationError as e:
                messages.append(
                    {
                        "role": "function",
                        "name": subfunction_name,
                        "content": json.dumps(
                            {"error": "Validation Failed", "details": e.message},
                            indent=4,
                        ),
                    }
                )
                continue
            except SchemaError as e:
                messages.append(
                    {
                        "role": "function",
                        "name": subfunction_name,
                        "content": json.dumps(
                            {"error": "Validation Failed", "details": e.message},
                            indent=4,
                        ),
                    }
                )
                continue

        if function_name == "search_care_pathways":
            queried_pathways = search(care_pathways, function_args.get("search_query"))

            queried_pathways = [{"name": d["name"]} for d in queried_pathways]

            messages.append(
                {
                    "role": "function",
                    "name": "search_care_pathways",
                    "content": json.dumps(queried_pathways, indent=4),
                }
            )
        elif function_name == "get_care_pathway":
            value = [d for d in care_pathways if d["name"] == function_args.get("name")]

            if len(value) == 0:
                value = "NO VALUE WITH THAT NAME FOUND"
            else:
                value = value[0]["content"]

            messages.append(
                {
                    "role": "function",
                    "name": "get_care_pathway",
                    "content": json.dumps({"value": value}, indent=4),
                }
            )

        elif function_name == "search_models":
            queried_functions = search(med_models, function_args.get("search_query"))

            messages.append(
                {
                    "role": "function",
                    "name": "search_models",
                    "content": json.dumps(queried_functions, indent=4),
                }
            )
        elif function_name == "search_diagnostic_tests":
            queried_functions = search(
                diagnostic_tests, function_args.get("search_query")
            )
            messages.append(
                {
                    "role": "function",
                    "name": "search_diagnostic_tests",
                    "content": json.dumps(queried_functions, indent=4),
                }
            )
        elif function_name == "list_ehr":
            only_updates = [
                {k: v for k, v in d.items() if k != "embedding"}
                for d in ehr
                if d.get("created")
            ]

            messages.append(
                {
                    "role": "function",
                    "name": "list_ehr",
                    "content": json.dumps(only_updates, indent=4),
                }
            )
        elif function_name == "search_values_in_ehr":
            queried_values = search(ehr, function_args.get("search_query"))
            messages.append(
                {
                    "role": "function",
                    "name": "search_values_in_ehr",
                    "content": json.dumps(queried_values, indent=4),
                }
            )
        elif function_name == "get_value_from_ehr":
            value = [d for d in ehr if d["name"] == function_args.get("name")]

            if len(value) == 0:
                value = "NO VALUE WITH THAT NAME FOUND"
            else:
                value = value[0]["content"]

            messages.append(
                {
                    "role": "function",
                    "name": "get_value_from_ehr",
                    "content": json.dumps({"value": value}, indent=4),
                }
            )
        elif function_name == "search_referral_clinics":
            queried_clinics = search(clinics, function_args.get("search_query"))
            messages.append(
                {
                    "role": "function",
                    "name": "search_referral_clinics",
                    "content": json.dumps(queried_clinics, indent=4),
                }
            )

        else:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            print(
                f"ChatGPT wants to use the function/model: {function_name} with arguments: {function_args}"
            )
            user_prompt = input("What should be returned?\n")
            match = re.search(r"INSERT (\w+)", user_prompt)

            print(match)

            if match:
                file = match.group(1)

                print(file)

                with open(f"data/{PATIENT}/{file}.md", "r") as file:
                    text = file.read()

                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": text,
                    }
                )
            else:
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": user_prompt,
                    }
                )
