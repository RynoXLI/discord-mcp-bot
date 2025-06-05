import os
import json
import logging
import discord
from discord.ext import commands
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from contextlib import asynccontextmanager
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from .config import load_config
from .llm import get_llm, get_prompt
from langchain_core.messages.utils import count_tokens_approximately

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
try:
    # Try to get system prompt file path from config first
    system_prompt_file = config.get("system_prompt", "prompts/default.txt")
    system_prompt = get_prompt(system_prompt_file)
except FileNotFoundError:
    system_prompt = "You are a discord bot, your job is to help users to the best of your ability. Please respond in discord markdown format. When responding to users, you have access to their Discord user IDs in the format 'Username (ID: 123456789)'. If you want to mention a specific user in your response, use the format <@123456789> to create a proper Discord mention."

intents = discord.Intents.default()
intents.message_content = True

# Use commands.Bot instead of discord.Client to support slash commands
bot = commands.Bot(command_prefix='$', intents=intents)


class ConversationManager:
    """
    Manages conversation context and LLM interactions for the Discord bot using an agent with MCP tools.
    """
    
    def __init__(
        self,
        llm,
        system_prompt,
        discord_client,
        mcp_servers=None,
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
            mcp_servers: Dictionary of MCP server configurations
            max_messages: Default maximum number of messages to retrieve
            max_time_window_minutes: Default time window for channel mode (in minutes)            max_tokens: Maximum tokens for LLM context trimming
        """
        
        self.llm = llm
        self.system_prompt = system_prompt
        self.discord_client = discord_client
        self.mcp_servers = mcp_servers or {}
        self.max_messages = max_messages
        self.max_time_window_minutes = max_time_window_minutes
        self.max_tokens = max_tokens
        
        # Create a trimmer for token management
        self.trimmer = trim_messages(
            strategy="last",
            max_tokens=max_tokens,
            include_system=True,
            token_counter=count_tokens_approximately,
        )

    @asynccontextmanager
    async def get_agent(self):
        """
        Create and yield an agent with MCP tools, applying tool filtering.
        
        Returns:
            Agent instance with filtered MCP tools
        """
        if self.mcp_servers:
            async with MultiServerMCPClient(self.mcp_servers) as mcp_client:
                # Get all available tools
                all_tools = mcp_client.get_tools()
                
                # Apply tool filtering for each server
                filtered_tools = []
                for tool in all_tools:
                    tool_name = tool.name
                    # Find which server this tool belongs to by checking tool configurations
                    should_include = True
                    
                    # Check each server's tool filtering configuration
                    for server_name, server_config in self.mcp_servers.items():
                        tool_config = server_config.get('tools', {})
                        if not tool_config:
                            continue
                            
                        mode = tool_config.get('mode', 'none')
                        
                        if mode == 'allow':
                            allowlist = tool_config.get('allowlist', [])
                            if tool_name not in allowlist:
                                should_include = False
                                break
                        elif mode == 'ban':
                            banlist = tool_config.get('banlist', [])
                            if tool_name in banlist:
                                should_include = False
                                break
                    
                    if should_include:
                        filtered_tools.append(tool)
                
                # print(f"MCP Tools loaded: {len(filtered_tools)}/{len(all_tools)} tools available")
                
                agent = create_react_agent(
                    model=self.llm,
                    tools=filtered_tools,
                    prompt=self.system_prompt,
                )
                yield agent
        else:
            # Fallback to simple chain if no MCP servers
            chain = self.trimmer | self.llm
            yield chain

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
        Get response from the LLM agent with MCP tools and token trimming.

        Args:
            messages: List of formatted messages for the LLM

        Returns:
            str: LLM response content
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
                    agent_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    agent_messages.append({"role": "assistant", "content": msg.content})

            async with self.get_agent() as agent:
                if self.mcp_servers:
                    # Use agent with MCP tools
                    response = await agent.ainvoke({"messages": agent_messages})
                    return response["messages"][-1].content
                else:
                    # Fallback to simple chain
                    response = await agent.ainvoke(trimmed_messages)
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
        summary = {
            "max_messages": self.max_messages,
            "max_time_window_minutes": self.max_time_window_minutes,
            "max_tokens": self.max_tokens,
            "has_token_trimming": self.trimmer is not None,
            "llm_class": type(self.llm).__name__ if self.llm else "None",
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt,
            "mcp_servers": list(self.mcp_servers.keys()) if self.mcp_servers else [],
        }
        
        # Add tool filtering information for each server
        tool_filtering_info = {}
        for server_name, server_config in self.mcp_servers.items():
            tool_config = server_config.get('tools', {})
            if tool_config:
                filter_mode = tool_config.get('mode', 'none')
                filter_info = {"mode": filter_mode}
                
                if filter_mode == 'allow':
                    filter_info["allowlist"] = tool_config.get('allowlist', [])
                elif filter_mode == 'ban':
                    filter_info["banlist"] = tool_config.get('banlist', [])
                    
                tool_filtering_info[server_name] = filter_info
                
        if tool_filtering_info:
            summary["tool_filtering"] = tool_filtering_info
        
        return summary


# Initialize the conversation manager with custom settings
conversation_manager = ConversationManager(
    llm=first_llm,
    system_prompt=system_prompt,
    discord_client=bot,
    mcp_servers=config.get('mcpServers', {}),  # Pass MCP servers configuration
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


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if this message is part of a reply chain
    if message.reference and message.reference.message_id:
        try:
            replied_message = await message.channel.fetch_message(
                message.reference.message_id
            )
            if replied_message.author == bot.user:
                # Get conversation formatted for LLM using replies mode
                formatted_messages = await conversation_manager.get_reply_conversation(
                    message
                )

                # Show typing indicator while getting LLM response
                async with message.channel.typing():
                    # Get LLM response
                    llm_response = await conversation_manager.get_llm_response(
                        formatted_messages
                    )                # Send response (potentially split into multiple messages)
                await send_long_response(message, llm_response)

                print(f"LLM Response: {llm_response}")
        except (discord.NotFound, discord.Forbidden):
            pass

    # Check if bot is mentioned, but don't double-reply if it's already handled in reply chain
    elif bot.user.mentioned_in(message):
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

    # Handle debug commands
    if message.content.startswith("$hello"):
        await message.channel.send("Hello!")
    elif message.content.startswith("$config"):
        config_summary = conversation_manager.get_config_summary()
        config_text = json.dumps(config_summary, indent=2)
        await message.channel.send(
            f"**Bot Configuration:**\n```json\n{config_text}\n```"
        )
    elif message.content.startswith("$tools"):
        # Show available tools (requires MCP servers to be configured)
        if conversation_manager.mcp_servers:
            try:
                # Get tools directly from the MCP client instead of from agent
                async with MultiServerMCPClient(conversation_manager.mcp_servers) as mcp_client:
                    all_tools = mcp_client.get_tools()
                    
                    # Apply the same filtering logic as in get_agent
                    filtered_tools = []
                    for tool in all_tools:
                        tool_name = tool.name
                        should_include = True
                        
                        # Check each server's tool filtering configuration
                        for server_name, server_config in conversation_manager.mcp_servers.items():
                            tool_config = server_config.get('tools', {})
                            if not tool_config:
                                continue
                                
                            mode = tool_config.get('mode', 'none')
                            
                            if mode == 'allow':
                                allowlist = tool_config.get('allowlist', [])
                                if tool_name not in allowlist:
                                    should_include = False
                                    break
                            elif mode == 'ban':
                                banlist = tool_config.get('banlist', [])
                                if tool_name in banlist:
                                    should_include = False
                                    break
                        
                        if should_include:
                            filtered_tools.append(tool)
                    
                    if filtered_tools:
                        tool_info = []
                        for tool in filtered_tools:
                            tool_name = tool.name
                            tool_description = getattr(tool, 'description', 'No description available')
                            tool_info.append(f"• **{tool_name}**: {tool_description}")
                        
                        tools_text = "\n".join(tool_info)
                          # Get filtering info from server configurations
                        filter_info = []
                        for server_name, server_config in conversation_manager.mcp_servers.items():
                            tool_config = server_config.get('tools', {})
                            if tool_config:
                                mode = tool_config.get('mode', 'none')
                                filter_info.append(f"{server_name}: {mode}")
                        
                        filter_summary = ", ".join(filter_info) if filter_info else "none"
                        response = f"**Available MCP Tools** (Filtering: {filter_summary}):\n{tools_text}\n\n**Total:** {len(filtered_tools)}/{len(all_tools)} tools available"
                        
                        if len(response) > 2000:
                            # Split long responses
                            await send_long_response(message, response)
                        else:
                            await message.channel.send(response)
                    else:
                        await message.channel.send("No tools available after filtering.")
            except Exception as e:
                await message.channel.send(f"Error getting tool information: {e}")
        else:
            await message.channel.send("No MCP servers configured. Tools are not available.")

    # Process commands (required for slash commands to work alongside message events)
    await bot.process_commands(message)


# Slash commands
@bot.slash_command(name="tools", description="Show available MCP tools with filtering information")
async def tools_slash(ctx):
    """Show available tools as a slash command"""
    await ctx.defer()  # Important for commands that might take time
    
    if conversation_manager.mcp_servers:
        try:
            # Get tools directly from the MCP client instead of from agent
            async with MultiServerMCPClient(conversation_manager.mcp_servers) as mcp_client:
                all_tools = mcp_client.get_tools()
                
                # Apply the same filtering logic as in get_agent
                filtered_tools = []
                for tool in all_tools:
                    tool_name = tool.name
                    should_include = True
                    
                    # Check each server's tool filtering configuration
                    for server_name, server_config in conversation_manager.mcp_servers.items():
                        tool_config = server_config.get('tools', {})
                        if not tool_config:
                            continue
                            
                        mode = tool_config.get('mode', 'none')
                        
                        if mode == 'allow':
                            allowlist = tool_config.get('allowlist', [])
                            if tool_name not in allowlist:
                                should_include = False
                                break
                        elif mode == 'ban':
                            banlist = tool_config.get('banlist', [])
                            if tool_name in banlist:
                                should_include = False
                                break
                    
                    if should_include:
                        filtered_tools.append(tool)
                
                if filtered_tools:
                    tool_info = []
                    for tool in filtered_tools:
                        tool_name = tool.name
                        tool_description = getattr(tool, 'description', 'No description available')
                        tool_info.append(f"• **{tool_name}**: {tool_description}")
                    
                    tools_text = "\n".join(tool_info)
                    
                    # Get filtering info from server configurations
                    filter_info = []
                    for server_name, server_config in conversation_manager.mcp_servers.items():
                        tool_config = server_config.get('tools', {})
                        if tool_config:
                            mode = tool_config.get('mode', 'none')
                            filter_info.append(f"{server_name}: {mode}")
                    
                    filter_summary = ", ".join(filter_info) if filter_info else "none"
                    response = f"**Available MCP Tools** (Filtering: {filter_summary}):\n{tools_text}\n\n**Total:** {len(filtered_tools)}/{len(all_tools)} tools available"
                    
                    if len(response) > 2000:
                        # Split long responses using followup for slash commands
                        await ctx.followup.send(response[:2000])
                        remaining = response[2000:]
                        while remaining:
                            chunk = remaining[:2000]
                            remaining = remaining[2000:]
                            await ctx.followup.send(chunk)
                    else:
                        await ctx.followup.send(response)
                else:
                    await ctx.followup.send("No tools available after filtering.")
        except Exception as e:
            await ctx.followup.send(f"Error getting tool information: {e}")
    else:
        await ctx.followup.send("No MCP servers configured. Tools are not available.")

@bot.slash_command(name="config", description="Show bot configuration")
async def config_slash(ctx):
    """Show bot configuration as a slash command"""
    await ctx.defer()
    
    config_summary = conversation_manager.get_config_summary()
    config_text = json.dumps(config_summary, indent=2)
    response = f"**Bot Configuration:**\n```json\n{config_text}\n```"
    
    if len(response) > 2000:
        await ctx.followup.send("**Bot Configuration:**\n```json")
        await ctx.followup.send(config_text[:1900] + "\n```")
        if len(config_text) > 1900:
            await ctx.followup.send("```json\n" + config_text[1900:] + "\n```")
    else:
        await ctx.followup.send(response)

@bot.slash_command(name="hello", description="Say hello!")
async def hello_slash(ctx):
    """Simple hello command as a slash command"""
    await ctx.respond("Hello! 👋")

@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands(register_guild_commands=True, guild_ids=[675538935472062479])
    print(f"{bot.user.name} connected.")


# Text-based commands (keeping for backwards compatibility)


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
