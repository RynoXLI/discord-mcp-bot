# filepath: s:\code\mcp-bot\discord-mcp-bot\src\discord_mcp_bot\bot.py
"""
Discord MCP Bot - Main Entry Point

This is the main bot file that initializes the Discord bot and sets up all components.
"""

import os
import logging
import discord
from discord.ext import commands
from dotenv import load_dotenv

from .config import load_config
from .llm import get_llm, get_prompt
from .conversation import ConversationManager
from .commands import BotCommands
from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name=os.getenv("PHOENIX_PROJECT_NAME", 'discord-bot'), # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config("configuration.yml")

# Get the LLM from configuration (new singular structure)
llm_config = config.get("llm", {})
first_llm = get_llm(llm_config) if llm_config else None

# Load system prompt from configuration or file
system_prompt_file = config.get("system_prompt", {}).get("file", "prompts/default.txt")
system_prompt = get_prompt(system_prompt_file)
system_prompt = system_prompt.strip() if system_prompt else "You are a helpful assistant."

intents = discord.Intents.default()
intents.message_content = True

# Use commands.Bot instead of discord.Client to support slash commands
bot = commands.Bot(command_prefix="$", intents=intents)

# Initialize the conversation manager with custom settings
conversation_manager = ConversationManager(
    llm=first_llm,
    system_prompt=system_prompt,
    discord_client=bot,
    mcp_servers=config.get("mcpServers", {}),  # Pass MCP servers configuration
    max_messages=config.get("max_recent_messages", 10),  # Allow more messages for better context
    max_time_window_minutes=config.get("max_recent_time_window_minutes", 30),  # Longer time window for channel mode
    max_tokens=config.get("max_tokens", 1000),  # Standard token limit for most models
)

# Print configuration summary on startup
print("Bot Configuration:")
print(
    f"  LLM: {llm_config.get('class', 'None')} ({llm_config.get('params', {}).get('model', 'unknown model')})"
)
print(f"  System Prompt: {config.get('system_prompt', 'default')}")
print(f"  MCP Servers: {list(config.get('mcpServers', {}).keys())}")
# print(f"  Conversation Manager: {conversation_manager.get_config_summary()}")

# Initialize bot commands
bot_commands = BotCommands(bot, conversation_manager, config)


def main():
    """Main function to start the Discord bot"""
    try:
        # Get the Discord token from configuration
        discord_token = os.getenv("DISCORD_TOKEN")
        if not discord_token:
            print("Discord token not found in configuration!")
            return

        print("Starting Discord MCP Bot...")
        bot.run(discord_token)
    except Exception as e:
        print(f"Failed to start bot: {e}")
        raise


if __name__ == "__main__":
    main()
