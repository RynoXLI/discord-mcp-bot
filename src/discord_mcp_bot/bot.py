import os
import json
import discord
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from .config import load_config
from .llm import get_llm, get_prompt
from langchain_core.messages.utils import count_tokens_approximately

load_dotenv()

# Load configuration
config = load_config("configuration.yml")

# Get the LLM from configuration (new singular structure)
llm_config = config.get("llm", {})
first_llm = get_llm(llm_config) if llm_config else None

# Load system prompt from configuration or file
try:
    # Try to get system prompt file path from config first
    system_prompt_file = config.get("system_prompt", "prompts/default.txt")
    system_prompt = get_prompt(system_prompt_file)
except FileNotFoundError:
    system_prompt = "You are a discord bot, your job is to help users to the best of your ability. Please respond in discord markdown format. When responding to users, you have access to their Discord user IDs in the format 'Username (ID: 123456789)'. If you want to mention a specific user in your response, use the format <@123456789> to create a proper Discord mention."

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


class ConversationManager:
    """
    Manages conversation context and LLM interactions for the Discord bot.
    """

    def __init__(
        self,
        llm,
        system_prompt,
        discord_client,
        max_messages=10,
        max_time_window_minutes=30,
        max_tokens=4096,
    ):
        """
        Initialize the ConversationManager.

        Args:
            llm: The LLM instance to use for responses
            system_prompt: The system prompt to use
            discord_client: The Discord client instance
            max_messages: Default maximum number of messages to retrieve
            max_time_window_minutes: Default time window for channel mode (in minutes)
            max_tokens: Maximum tokens for LLM context trimming
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.discord_client = discord_client
        self.max_messages = max_messages
        self.max_time_window_minutes = max_time_window_minutes
        self.max_tokens = max_tokens

        # Trim messages to avoid exceeding token limits
        trimmer = trim_messages(
            strategy="last",
            max_tokens=max_tokens,
            include_system=True,
            token_counter=count_tokens_approximately,
        )
        self.chain = trimmer | llm

    async def get_reply_chain(self, message, max_messages=None):
        """
        Get the reply chain for a message by traversing backwards.

        Args:
            message: The Discord message to start from
            max_messages: Maximum number of messages to retrieve (uses instance default if None)

        Returns:
            List of Discord messages in chronological order
        """
        if max_messages is None:
            max_messages = self.max_messages

        current_message = message
        chain = []
        depth = 0

        # Traverse backwards to find the root of the reply chain
        while current_message and depth < max_messages:
            chain.insert(0, current_message)  # Insert at beginning to maintain order
            depth += 1

            if current_message.reference and current_message.reference.message_id:
                try:
                    current_message = await current_message.channel.fetch_message(
                        current_message.reference.message_id
                    )
                except (discord.NotFound, discord.Forbidden):
                    break
            else:
                break

        return chain

    async def get_recent_channel_messages(
        self, message, max_messages=None, time_window_minutes=None
    ):
        """
        Get recent channel messages within a time window.

        Args:
            message: The Discord message to start from
            max_messages: Maximum number of messages to retrieve (uses instance default if None)
            time_window_minutes: Only look at messages from the past X minutes (uses instance default if None)

        Returns:
            List of Discord messages in chronological order
        """
        if max_messages is None:
            max_messages = self.max_messages
        if time_window_minutes is None:
            time_window_minutes = self.max_time_window_minutes

        chain = []
        cutoff_time = datetime.now(message.created_at.tzinfo) - timedelta(
            minutes=time_window_minutes
        )

        try:
            # Get recent messages from the channel (including the current message)
            message_count = 0
            async for hist_msg in message.channel.history(
                limit=50, before=message
            ):  # Increased limit to account for filtering
                # Only include messages within the time window
                if hist_msg.created_at >= cutoff_time:
                    chain.insert(
                        0, hist_msg
                    )  # Insert at beginning to maintain chronological order
                    message_count += 1
                    if message_count >= max_messages:
                        break
                else:
                    # Stop if we've gone beyond the time window
                    break

            # Add the current message at the end
            chain.append(message)
        except (discord.Forbidden, discord.HTTPException):
            # If we can't read history, just use the current message
            chain = [message]

        return chain

    def format_messages_for_llm(self, discord_messages):
        """
        Convert Discord messages to LangChain message objects.

        Args:
            discord_messages: List of Discord message objects

        Returns:
            List of LangChain message objects formatted for the LLM
        """
        # Start with system prompt
        messages = [SystemMessage(content=self.system_prompt)]

        # Convert Discord messages to LangChain message objects
        for discord_msg in discord_messages:
            username = discord_msg.author.display_name or discord_msg.author.name
            user_id = discord_msg.author.id
            content = discord_msg.content

            if (
                username == self.discord_client.user.display_name
                or username == self.discord_client.user.name
            ):
                messages.append(AIMessage(content=content))
            else:
                # Include user ID so the bot can @ users appropriately
                formatted_content = f"{username} (ID: {user_id}): {content}"
                messages.append(HumanMessage(content=formatted_content))

        return messages

    async def get_conversation_for_llm(
        self, message, mode="replies", max_messages=None, time_window_minutes=None
    ):
        """
        Get conversation context and format it for LLM processing.

        Args:
            message: The Discord message to start from
            mode: "replies" for reply chain context, "channel" for recent channel messages
            max_messages: Maximum number of messages to retrieve (uses instance default if None)
            time_window_minutes: For channel mode, only look at messages from the past X minutes (uses instance default if None)

        Returns:
            List of LangChain message objects formatted for the LLM
        """
        # Use instance defaults if not provided
        if max_messages is None:
            max_messages = self.max_messages
        if time_window_minutes is None:
            time_window_minutes = self.max_time_window_minutes

        if mode == "replies":
            msgs = await self.get_reply_chain(message, max_messages)
        elif mode == "channel":
            msgs = await self.get_recent_channel_messages(
                message, max_messages, time_window_minutes
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'replies' or 'channel'")

        return self.format_messages_for_llm(msgs)

    async def get_llm_response(self, messages):
        """
        Get response from the LLM with token trimming.

        Args:
            messages: List of formatted messages for the LLM

        Returns:
            str: LLM response content
        """
        try:
            response = await self.chain.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    async def get_reply_conversation(self, message):
        """
        Convenience method to get reply chain conversation using instance defaults.

        Args:
            message: The Discord message to start from

        Returns:
            List of LangChain message objects formatted for the LLM
        """
        return await self.get_conversation_for_llm(message, mode="replies")

    async def get_channel_conversation(self, message):
        """
        Convenience method to get channel conversation using instance defaults.

        Args:
            message: The Discord message to start from

        Returns:
            List of LangChain message objects formatted for the LLM
        """
        return await self.get_conversation_for_llm(message, mode="channel")

    def get_config_summary(self):
        """
        Get a summary of the conversation manager configuration.

        Returns:
            dict: Configuration summary
        """
        return {
            "max_messages": self.max_messages,
            "max_time_window_minutes": self.max_time_window_minutes,
            "max_tokens": self.max_tokens,
            "has_token_trimming": self.chain is not None,
            "llm-class": type(self.llm).__name__ if self.llm else "None",
            "system_prompt": self.system_prompt,
        }


# Initialize the conversation manager with custom settings
conversation_manager = ConversationManager(
    llm=first_llm,
    system_prompt=system_prompt,
    discord_client=client,
    max_messages=15,  # Allow more messages for better context
    max_time_window_minutes=45,  # Longer time window for channel mode
    max_tokens=4096,  # Standard token limit for most models
)

# Print configuration summary on startup
print("Bot Configuration:")
print(
    f"  LLM: {llm_config.get('class', 'None')} ({llm_config.get('params', {}).get('model', 'unknown model')})"
)
print(f"  System Prompt: {config.get('system_prompt', 'default')}")
print(f"  MCP Servers: {list(config.get('mcpServers', {}).keys())}")
print(f"  Conversation Manager: {conversation_manager.get_config_summary()}")


async def send_long_response(message, response_text, max_length=2000):
    """
    Send a long response, splitting it into multiple messages if necessary.

    Args:
        message: The Discord message to reply to
        response_text: The response text to send
        max_length: Maximum length per message (default: 2000)

    Returns:
        The last message sent (for potential replies)
    """
    if len(response_text) <= max_length:
        return await message.reply(response_text)

    # Split the response into chunks
    chunks = []
    current_chunk = ""

    # Split by lines first to avoid breaking mid-sentence
    lines = response_text.split("\n")

    for line in lines:
        # If adding this line would exceed the limit
        if len(current_chunk + line + "\n") > max_length:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = line + "\n"
            else:
                # Single line is too long, split it by words
                words = line.split(" ")
                for word in words:
                    if len(current_chunk + word + " ") > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.rstrip())
                            current_chunk = word + " "
                        else:
                            # Single word is too long, just add it anyway
                            chunks.append(word)
                            current_chunk = ""
                    else:
                        current_chunk += word + " "
                current_chunk += "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.rstrip())

    # Send the chunks
    last_message = None
    for i, chunk in enumerate(chunks):
        if i == 0:
            # Reply to the original message
            last_message = await message.reply(chunk)
        else:
            # Reply to the previous bot message to create a chain
            last_message = await last_message.reply(chunk)

    return last_message


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if this message is part of a reply chain
    if message.reference and message.reference.message_id:
        try:
            replied_message = await message.channel.fetch_message(
                message.reference.message_id
            )
            if replied_message.author == client.user:
                # Get conversation formatted for LLM using replies mode
                formatted_messages = await conversation_manager.get_reply_conversation(
                    message
                )

                # Show typing indicator while getting LLM response
                async with message.channel.typing():
                    # Get LLM response
                    llm_response = await conversation_manager.get_llm_response(
                        formatted_messages
                    )

                # Send response (potentially split into multiple messages)
                await send_long_response(message, llm_response)

                print(f"LLM Response: {llm_response}")
        except (discord.NotFound, discord.Forbidden):
            pass

    # Check if bot is mentioned, but don't double-reply if it's already handled in reply chain
    elif client.user.mentioned_in(message):
        # Get conversation formatted for LLM using channel mode for recent context
        formatted_messages = await conversation_manager.get_channel_conversation(
            message
        )
        print(f"Channel conversation context: {len(formatted_messages)} messages")

        # Show typing indicator while getting LLM response
        async with message.channel.typing():
            # Get LLM response
            llm_response = await conversation_manager.get_llm_response(
                formatted_messages
            )

        # Send response (potentially split into multiple messages)
        await send_long_response(message, llm_response)

    if message.content.startswith("$hello"):
        await message.channel.send("Hello!")
    elif message.content.startswith("$config"):
        config_summary = conversation_manager.get_config_summary()
        config_text = json.dumps(config_summary, indent=2)
        await message.channel.send(
            f"**Bot Configuration:**\n```json\n{config_text}\n```"
        )


def main():
    client.run(os.getenv("DISCORD_TOKEN"))
