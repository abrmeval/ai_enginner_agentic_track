import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from IPython.display import Markdown, display
import gradio as gr
import json
import requests

load_dotenv(override=True)

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


def get_profile():
    pdf_path = os.path.join(current_file_dir, "..", "me", "profile.pdf")
    reader = PdfReader(pdf_path)
    linkedin = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text
    return linkedin


def get_summary():
    summary_path = os.path.join(current_file_dir, "..", "me", "summary.txt")
    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


def get_system_prompt(summary, linkedin):
    return f"""

# Your role

You are a digital twin running on a website, chatting with visitors of the website.
You represent the person who's website you are on.
You answer questions related to their career, background, education, skills and experience.

Here are the details of the person you are representing:

<summarty>
{summary}
</summary>

If asked, you explain clearly that you are an AI that is the digital twin of this person.

# Context

Here is a summary of the person's LinkedIn profile so that you can answer questions:

<linkedin_profile>
{linkedin}
</linkedin_profile>

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

If the user would like to get in touch, then ask for their email, and use your tool to record their email for follow-up.

IMPORTANT: You MUST only answer questions related to their carrer, background and everything based on the summary and linkedin profile.
If you don't know the answer, say so. Never make up an answer.
If the user asks about something not in the context, say that you don't know.
At the end of every of your response, at the following legend: "My knowledge is limited and I can only answer questions related to <person_name_from_linkedin> linkedin profile and summary."
"""


def get_provider_credentials(providers, provider_name):
    provider = providers.get(provider_name, None)
    false = False, "", ""
    if provider:
        endpoint = provider.get("endpoint", "")
        api_key = provider.get("api_key", "")
        if endpoint == "" or api_key == "":
            return false
        return True, endpoint, api_key
    return false


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return "OK"


def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return "OK"


tool_func = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}")

        # Call the corresponding tool function based on the name
        tool = tool_func.get(tool_name)
        result = tool(**arguments) if tool else "No tool found"
        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
    return results


def review_twin_response(assistant_response, history):
    messages = [
        {
            "role": "system",
            "content": """
            Your role is to review the AI digital twin response.
            You need to review if the response meets the following guidelines:
            - The digital twin MUST always mention at the begining of a conversation that it is a digital twin representing the person from the linkedin profile
            - The digital twin must show interest
            - It should mention at the end that its knowledge is limited and it can only respond based on the linkedin profile and summary given
            
            Respond always with a JSON contaning the boolean variable called is_ok, indicating True or False and another string variable called suggestions with the suggestions to enhance the response.
            Example: {"is_ok": true, "suggestions": "<your_suggestions_here>"}.
            """,
        },
        {
            "role": "user",
            "content": f"""The digital twin response is: {assistant_response}
                            The following is the summary and linkedin profile from the person the digital twin is representing:
                            <summary>{summary}</summary>
                            <profile>{profile}</profile>
                            <conversation_history>{history}</conversation_history>
                            """,
        },
    ]

    response = groq.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages
    )

    print(f"review_twin_response: {response.choices[0].message.content}")
    json_data = json.loads(response.choices[0].message.content)
    print(f"is_ok: {json_data}")
    return json_data


def needs_review(user_message):
    messages = [
        {
            "role": "system",
            "content": """
            Your role is to verify if the user is requesting information that a recruiter might ask a candidate for a job application.
            Read the message and verify if it indicates that the user is asking for information like experience, career, skills, background, education or any other related information that a recruiter would ask to a candidate.
            Steps to follow:
            - Review carefully the user's message
            - If the the user is GRETTING or ASKING for information that a recruiter would ask, return a JSON with a boolean variable called needs_review equals to True
            - If the user is asking for something else or nothing related to the candidate's profile or or professional information, return a JSON with a boolean variable called needs_review equals to False
            Note: Respond always with a JSON contaning the boolean variable called needs_review, indicating True or False.
            Example: {"needs_review": true} or {"needs_review": false}.""",
        },
        {
            "role": "user",
            "content": f"""The user message is following: 
                        <user_message>{user_message}</user_message>""",
        },
    ]

    response = openrouter.chat.completions.create(
        model="openrouter/free", messages=messages
    )

    json_data = json.loads(response.choices[0].message.content)
    print(f"needs_review: {json_data}")
    return json_data


def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "record_user_details",
                "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email address of this user",
                        },
                        "name": {
                            "type": "string",
                            "description": "The user's name, if they provided it",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Any additional info about the conversation that's worth recording to give context",
                        },
                    },
                    "required": ["email"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_unknown_question",
                "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question that couldn't be answered",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": message}]
    )
    tools = get_tools()
    tool_was_called = False

    while True:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b", messages=messages, tools=tools
        )

        # If the agent is going to call tools
        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            messages.append(message)

            # Calling the tools
            results = handle_tool_calls(message.tool_calls)
            # We append the result to thew message list
            messages.extend(results)

            # Call the AI model
            response = groq.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tools,
                max_completion_tokens=5000,
            )
            tool_was_called = True

        assistant_response = response.choices[0].message.content

        # If a tool was not called, we validate the user message (It Could change if we add tools to retrieve information)
        if not tool_was_called:
            needs_review_data = needs_review(message)

            if not needs_review_data.get("needs_review", False):
                break

            # Review the twin response and get suggestions if needed
            json_data = review_twin_response(assistant_response, json.dumps(messages))

            if json_data.get("is_ok", False):
                break

            messages.append(
                [
                    {
                        "role": "user",
                        "content": "The AI agent Response Reviewer suggest the following for your response: "
                        + json_data.get("suggestions", "No suggestions provided"),
                    }
                ]
            )
    return assistant_response


# AI Providers
providers = {
    "foundry": {
        "endpoint": os.getenv("AZURE_FOUNDRY_ENDPOINT"),
        "api_key": os.getenv("AZURE_FOUNDRY_API_KEY"),
    },
    "groq": {
        "endpoint": os.getenv("GROQ_ENDPOINT"),
        "api_key": os.getenv("GROQ_API_KEY"),
    },
    "openrouter": {
        "endpoint": os.getenv("OPENROUTER_ENDPOINT"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
}

profile = get_profile()
summary = get_summary()
system_prompt = get_system_prompt(summary, profile)

# Digital Twin
ok, endpoint, api_key = get_provider_credentials(providers, "groq")
groq = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")

# Digital Twin Response Reviewer
ok, endpoint, api_key = get_provider_credentials(providers, "foundry")
foundry = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")

# User Message Validator - > Validates if the user message requests context related information
ok, endpoint, api_key = get_provider_credentials(providers, "openrouter")
openrouter = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    if pushover_user.startswith("u"):
        print("Pushover user found and looks good")
    else:
        print("Pushover user found but doesn't start with u")
else:
    print("Pushover user not found")

if pushover_token:
    if pushover_token.startswith("a"):
        print("Pushover token found and looks good")
    else:
        print("Pushover token found but doesn't start with a")
else:
    print("Pushover token not found")

# Launch the Web UI
gr.ChatInterface(chat).launch(inbrowser=True)
