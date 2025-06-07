"""
Conversation management for Discord MCP Bot.

This module handles conversation context, message formatting, and LLM interactions.
"""

import logging
import re
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import discord
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation context and LLM interactions for the Discord bot using an agent with MCP tools.
    """

    def __init__(
        self,
        llm,
        system_prompt: str,
        discord_client,
        mcp_servers: Optional[dict] = None,
        max_messages: int = 10,
        max_time_window_minutes: int = 30,
        max_tokens: int = 4096,
    ):
        """
        Initialize the ConversationManager.

        Args:
            llm: The LLM instance to use for responses
            system_prompt: The system prompt to use
            discord_client: The Discord client instance
            mcp_servers: Dictionary of MCP server configurations
            max_messages: Default maximum number of messages to retrieve
            max_time_window_minutes: Default time window for channel mode (in minutes)
            max_tokens: Maximum tokens for LLM context trimming
        """

        self.llm = llm
        self.system_prompt = system_prompt
        self.discord_client = discord_client
        self.mcp_servers = mcp_servers or {}
        self.max_messages = max_messages
        self.max_time_window_minutes = max_time_window_minutes
        self.max_tokens = max_tokens

        # Check if LLM supports tools
        self.tools_supported = self._check_tools_support()
        if not self.tools_supported:
            logger.warning(
                f"LLM {type(self.llm).__name__} does not support bind_tools. MCP tools will be disabled."
            )
            self.mcp_servers = {}  # Disable MCP servers if tools aren't supported
            
        # Create a trimmer for token management
        self.trimmer = trim_messages(
            strategy="last",
            max_tokens=max_tokens,
            include_system=True,
            token_counter=count_tokens_approximately,
        )

    def _check_tools_support(self) -> bool:
        """
        Check if the LLM supports bind_tools method.
        
        Returns:
            bool: True if the LLM supports tools, False otherwise
        """
        return hasattr(self.llm, 'bind_tools') and callable(getattr(self.llm, 'bind_tools', None))

    @asynccontextmanager
    async def get_agent(self):
        """
        Create and yield an agent with MCP tools, applying tool filtering.

        Returns:
            Agent instance with filtered MCP tools or simple chain if tools not supported
        """
        if self.mcp_servers and self.tools_supported:
            async with MultiServerMCPClient(self.mcp_servers) as mcp_client:
                # Get all available tools
                all_tools = mcp_client.get_tools()
                self.all_tools = all_tools  # Store for later use

                # Apply tool filtering for each server
                filtered_tools = []
                for tool in all_tools:
                    tool_name = tool.name
                    # Find which server this tool belongs to by checking tool configurations
                    should_include = True

                    # Check each server's tool filtering configuration
                    for server_name, server_config in self.mcp_servers.items():
                        tool.name = f"{server_name}_{tool.name}"
                        print(tool_name)
                        tool_config = server_config.get("tools", {})
                        if not tool_config:
                            continue

                        mode = tool_config.get("mode", "none")
                        tool_list = tool_config.get("list", [])

                        if mode == "allow":
                            if tool_name not in tool_list:
                                should_include = False
                                break
                        elif mode == "ban":
                            if tool_name in tool_list:
                                should_include = False
                                break

                    if should_include:
                        filtered_tools.append(tool)

                agent = create_react_agent(
                    model=self.llm,
                    tools=filtered_tools,
                    prompt=self.system_prompt,
                )
                self.tools = filtered_tools
                yield agent
        else:
            # Fallback to simple chain if no MCP servers or tools not supported
            chain = self.trimmer | self.llm
            yield chain

    async def get_reply_chain(
        self, message: discord.Message, max_messages: Optional[int] = None
    ) -> List[discord.Message]:
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
        self,
        message: discord.Message,
        max_messages: Optional[int] = None,
        time_window_minutes: Optional[int] = None,
    ) -> List[discord.Message]:
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

    def format_messages_for_llm(self, discord_messages: List[discord.Message]) -> List:
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
        self,
        message: discord.Message,
        mode: str = "replies",
        max_messages: Optional[int] = None,
        time_window_minutes: Optional[int] = None,
    ) -> List:
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

    async def get_llm_response(self, messages: List) -> Tuple[str, Optional[str]]:
        """
        Get response from the LLM agent with MCP tools and token trimming.

        Args:
            messages: List of formatted messages for the LLM

        Returns:
            Tuple of (clean_response, thoughts) where thoughts is None if no thinking found
        """
        try:
            # Trim messages to avoid token limits
            trimmed_messages = self.trimmer.invoke(messages)

            # Convert to the format expected by the agent
            agent_messages = []
            for msg in trimmed_messages:
                if isinstance(msg, SystemMessage):
                    # System message is handled by the agent's prompt
                    continue
                elif isinstance(msg, HumanMessage):
                    # Extract username from the formatted content for the role
                    content = msg.content
                    print(content)
                    if ":" in content and "(ID:" in content:
                        # Extract username from "Username (ID: 123456789): message"
                        username_part = content.split(":", 1)[0]
                        if "(ID:" in username_part:
                            username = username_part.split("(ID:")[0].strip()
                            print(content)
                            agent_messages.append(
                                {"role": username, "content": content}
                            )
                        else:
                            agent_messages.append({"role": "user", "content": content})
                    else:
                        agent_messages.append({"role": "user", "content": content})
                elif isinstance(msg, AIMessage):
                    agent_messages.append({"role": "assistant", "content": msg.content})
                    
            async with self.get_agent() as agent:
                if self.mcp_servers and self.tools_supported:
                    # Use agent with MCP tools
                    response = await agent.ainvoke({"messages": agent_messages})
                    raw_response = response["messages"][-1].content
                else:
                    # Fallback to simple chain
                    response = await agent.ainvoke(trimmed_messages)
                    raw_response = response.content
                
                # Extract thoughts from the response
                clean_response, thoughts = self.extract_thoughts(raw_response)
                return clean_response, thoughts

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "I'm sorry, I encountered an error while processing your request.", None
        
    def get_config_summary(self) -> dict:
        """
        Get a summary of the conversation manager configuration.

        Returns:
            dict: Configuration summary
        """
        summary = {
            "max_messages": self.max_messages,
            "max_time_window_minutes": self.max_time_window_minutes,
            "max_tokens": self.max_tokens,
            "has_token_trimming": self.trimmer is not None,
            "llm_class": type(self.llm).__name__ if self.llm else "None",
            "system_prompt": self.system_prompt
            if len(self.system_prompt) > 100
            else self.system_prompt,
            "mcp_servers": list(self.mcp_servers.keys()) if self.mcp_servers else [],
            "tools_supported": self.tools_supported,
            "tools_filtering": [tool.name for tool in self.tools] if self.tools_supported else [],
        }

        return summary

    def extract_thoughts(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking process from LLM response if present.

        Args:
            response: The full LLM response containing potential <think> tags

        Returns:
            Tuple of (clean_response, thoughts) where thoughts is None if no thinking found
        """
        # Pattern to match <think>...</think> tags (case insensitive, multiline)
        think_pattern = r"<think>(.*?)</think>"

        # Find all thinking blocks
        thoughts_matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)

        if thoughts_matches:
            # Combine all thinking blocks
            thoughts = "\n\n".join(match.strip() for match in thoughts_matches)

            # Remove all thinking blocks from the response
            clean_response = re.sub(think_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)

            # Clean up extra whitespace and newlines
            clean_response = re.sub(r"\n\s*\n\s*\n", "\n\n", clean_response.strip())

            return clean_response, thoughts

        return response, None