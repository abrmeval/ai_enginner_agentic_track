import gradio as gr
import context
from digital_twin import DigitalTwin
from providers import AiProvider
import agents.context as agents_context
from agents.agents import TwinResponseReviewer, UserMessageValidator
from tools import AiTools
from notifications import NotificationClient
import json


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = history + [{"role": "user", "content": message}]
    tool_was_called = False
    tools = ai_tools.get_tools()

    while True:
        response = digital_twin.prompt_agent(messages, tools)

        # If the agent is going to call tools
        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            messages.append(message)

            # Calling the tools
            results = ai_tools.handle_tool_calls(message.tool_calls)
            # We append the result to the message list
            messages.extend(results)

            # Call the AI model
            response = digital_twin.prompt_agent(messages, tools)
            tool_was_called = True

        assistant_response = response.choices[0].message.content

        # If a tool was not called, we validate the Twin response (It Could change if we add tools to retrieve information)
        if not tool_was_called:
            user_prompt = agents_context.get_validator_user_prompt(message)
            needs_review = message_validator.validate_user_message(user_prompt, tools)

            if not needs_review.get("review_twin_response", False):
                break

            # Review the twin response and get suggestions if needed
            user_prompt = agents_context.get_twin_reviewer_user_prompt(
                ai_context.get("summary"),
                ai_context.get("profile"),
                assistant_response,
                json.dumps(messages),
            )
            review = response_reviewer.get_review([user_prompt], tools)

            if review.get("is_ok", False):
                break

            messages.append(
                [
                    {
                        "role": "user",
                        "content": "The AI agent Response Reviewer suggest the following for your response: "
                        + review.get("suggestions", "No suggestions provided"),
                    }
                ]
            )
        tool_was_called = False

    return assistant_response


provider = AiProvider()
notification_client = NotificationClient()

ai_tools = AiTools(notification_client)
ai_context = context.get_ai_context()

# Digital Twin
digital_twin = DigitalTwin(provider, ai_context)

# Digital Twin Response Reviewer
response_reviewer = TwinResponseReviewer(
    provider, agents_context.get_twin_reviewer_instructions()
)

# User Message Validator
message_validator = UserMessageValidator(
    provider, agents_context.get_validator_instructions()
)

# Launch the Web UI
gr.ChatInterface(chat).launch(inbrowser=True)
