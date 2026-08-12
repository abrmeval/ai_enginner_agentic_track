from openai import OpenAI
import json


class TwinResponseReviewer:
    def __init__(self, provider, instructions):
        ok, endpoint, api_key = provider.get_provider_credentials("foundry")
        if not ok:
            raise Exception("Provider not found")

        self.client = OpenAI(base_url=endpoint, api_key=api_key)
        self.instructions = instructions

    def get_review(self, message, tools):
        message_list = [self.instructions, message]

        response = self.client.chat.completions.create(
            model="gpt-5-nano",
            messages=message_list,
            tools=tools,
            max_completion_tokens=5000,
        )

        json_data = json.loads(response.choices[0].message.content)
        return json_data


class UserMessageValidator:
    def __init__(self, provider, instructions):
        ok, endpoint, api_key = provider.get_provider_credentials("openrouter")
        if not ok:
            raise Exception("Provider not found")

        self.client = OpenAI(base_url=endpoint, api_key=api_key)
        self.instructions = instructions

    def validate_user_message(self, message, tools):
        message_list = [self.instructions, message]

        response = self.client.chat.completions.create(
            model="openrouter/free",
            messages=message_list,
            tools=tools,
            max_completion_tokens=5000,
        )
        json_data = json.loads(response.choices[0].message.content)
        return json_data
