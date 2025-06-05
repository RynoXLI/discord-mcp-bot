"""
Utility functions for Discord MCP Bot.

This module contains utility functions for configuration management, message handling, and MCP tools.
"""

import os
import yaml
import logging
from typing import List, Optional

import discord
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import load_config

logger = logging.getLogger(__name__)


async def send_long_response(message: discord.Message, response_text: str, max_length: int = 2000) -> Optional[discord.Message]:
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


def save_config(config_data: dict, file_path: str = "configuration.yml") -> None:
    """
    Save configuration back to YAML file.
    
    Args:
        config_data: Configuration data to save
        file_path: Path to configuration file
    """
    with open(file_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False, indent=2)


async def get_available_tools(mcp_servers: dict) -> List[str]:
    """
    Get all available tools from MCP servers.
    
    Args:
        mcp_servers: Dictionary of MCP server configurations
        
    Returns:
        List of tool names
    """
    if not mcp_servers:
        return []
    
    try:
        async with MultiServerMCPClient(mcp_servers) as mcp_client:
            return [tool.name for tool in mcp_client.get_tools()]
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        return []


def reload_conversation_manager(conversation_manager, config_file: str = "configuration.yml") -> None:
    """
    Reload the conversation manager with updated config.
    
    Args:
        conversation_manager: The conversation manager instance to update
        config_file: Path to configuration file
    """
    # Reload config from file
    config = load_config(config_file)
    conversation_manager.mcp_servers = config.get('mcpServers', {})


def print_startup_info(config: dict, conversation_manager) -> None:
    """
    Print bot configuration summary on startup.
    
    Args:
        config: Configuration dictionary
        conversation_manager: ConversationManager instance
    """
    llm_config = config.get("llm", {})
    
    print("Bot Configuration:")
    print(
        f"  LLM: {llm_config.get('class', 'None')} ({llm_config.get('params', {}).get('model', 'unknown model')})"
    )
    print(f"  System Prompt: {config.get('system_prompt', 'default')}")
    print(f"  MCP Servers: {list(config.get('mcpServers', {}).keys())}")
    print(f"  Conversation Manager: {conversation_manager.get_config_summary()}")


def get_discord_token() -> str:
    """
    Get Discord token from environment variables.
    
    Returns:
        Discord token string
        
    Raises:
        ValueError: If token is not found
    """
    discord_token = os.getenv("DISCORD_TOKEN")
    if not discord_token:
        raise ValueError("Discord token not found in configuration!")
    return discord_token
