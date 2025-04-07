# app/services/query_generation/prompts/summarizer_prompts.py

CONVERSATION_SUMMARY_PROMPT_TEMPLATE = """
Below is a conversation between a user and an assistant about database queries.
Please provide a concise summary (max 150 words) that captures the key topics and information.
The summary should help provide context for continuing the conversation.

CONVERSATION:
{conversation_text}

SUMMARY:
"""