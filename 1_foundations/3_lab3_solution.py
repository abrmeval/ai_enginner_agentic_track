import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from IPython.display import Markdown, display
import gradio as gr
import json

load_dotenv(override=True)

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_profile():
    pdf_path = os.path.join(script_dir, "me", "profile.pdf")
    reader = PdfReader(pdf_path)
    linkedin = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text
    return linkedin


def get_summary():
    summary_path = os.path.join(script_dir, "me", "summary.txt")
    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


def get_system_prompt(summary, linkedin):
    return f"""

# Your role

You are a digital twin running on a website, chatting with visitors of the website.
You represent the person who's website you are on.
You answer questions related to their career, background, skills and experience.

Here are the details of the person you are representing:

{summary}

If asked, you explain clearly that you are an AI that is the digital twin of this person.

# Context

Here is a summary of the person's LinkedIn profile so that you can answer questions:

{linkedin}

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

IMPORTANT: If you don't know the answer, say so. Never make up an answer.
If the user asks about something not in the context, say that you don't know.
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


def record_email_tool(email):
    print(f"Tool called to record an email: {email}")
    with open("emails.txt", "a", encoding="utf-8") as f:
        f.write(email + "\n")
    return "Email received"


def review_prompt(assistant_response, history):
    messages = [
        {
            "role": "system",
            "content": """
            Your role is to review the AI digital twin response.
            You need to review if the response follows the following guideliness:
            - The digital twin MUST always mention that it is a digital twin representing the person from the linkedin profile 
              when starting a new conversation
            - The digital twin must show interest
            - It should mention that has knowledge limitations and can only respond based on the linkedin profile and summary
            
            Respond always with a JSON contaning the boolean variable called is_ok, indicating True or False and another string variable called suggestions with the suggestions to enhance the response.
            Example: {"is_ok":true, "suggestions":"<your_suggestions_here>"}.
            """,
        },
        {
            "role": "user",
            "content": f"""The digital twin response is: {assistant_response}
                            The following is the summary and linkedin profile from the person the digital twin is representing:
                            SUMMARY: {summary}
                            PROFILE: {profile}
                            Digital twin conversation history:
                            {history}
                            """,
        },
    ]

    response = groq.chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages
    )

    print(f"review_prompt: {response.choices[0].message.content}")
    json_data = json.loads(response.choices[0].message.content)
    print(f"is_ok: {json_data}")
    return json_data


def needs_review(user_message):
    messages = [
        {
            "role": "system",
            "content": """
            Your role is to verify if the user is a recruiter who is asking for information about a candidate.
            Read the message and verify if it indicates that the user is asking for information like experience, career, skills, background, education or any other related information that a recruiter would ask to a candidate.
            Steps to follow:
            - Review carefully the user's message
            - If the the user is asking  for information that a recruiter would ask, return a JSON with a boolean variable called needs_review equals to True
            - If the user is asking for something else or nothing related to the candidate's profile or or professional information, return a JSON with a boolean variable called needs_review equals to False
            Note: Respond always with a JSON contaning the boolean variable called needs_review, indicating True or False.
            Example: {"needs_review":true} or {"needs_review":false}."""
        },
        {
            "role": "user",
            "content": f"""The user message is: {user_message}""",
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
                "name": "record_email_tool",
                "description": "Use this tool to record that a user provided their email address",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email address of this user",
                        }
                    },
                    "required": ["email"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": message}]
    )
    tools = get_tools()

    while True:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b", messages=messages, tools=tools
        )

        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            messages.append(message)
            for tool_call in message.tool_calls:
                email = json.loads(tool_call.function.arguments).get("email")
                record_email_tool(email)
                messages.append(
                    {
                        "role": "tool",
                        "content": "Email recorded",
                        "tool_call_id": tool_call.id,
                    }
                )
            response = groq.chat.completions.create(
                model="openai/gpt-oss-120b", messages=messages, tools=tools
            )

        assistant_response = response.choices[0].message.content
        needs_review_data = needs_review(message)

        if not needs_review_data.get("needs_review", False):
            break

        json_data = review_prompt(assistant_response, json.dumps(messages))

        if json_data.get("is_ok", False):
            break

        messages.append([{"role": "user", "content": "The AI agent reviewer suggest the following for your response: " + json_data.get("suggestions", "No suggestions provided")}])
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

# Twin
ok, endpoint, api_key = get_provider_credentials(providers, "groq")
groq = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")

# AI Response Reviewer
ok, endpoint, api_key = get_provider_credentials(providers, "foundry")
foundry = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")

# AI Message Reviewer
ok, endpoint, api_key = get_provider_credentials(providers, "openrouter")
openrouter = OpenAI(base_url=endpoint, api_key=api_key)

if not ok:
    raise Exception("Provider not found")


gr.ChatInterface(chat).launch(inbrowser=True)
