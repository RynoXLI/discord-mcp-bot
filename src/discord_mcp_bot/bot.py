import os
import json
import logging
import yaml
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
                        tool_list = tool_config.get('list', [])
                        
                        if mode == 'allow':
                            if tool_name not in tool_list:
                                should_include = False
                                break
                        elif mode == 'ban':
                            if tool_name in tool_list:
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
                
                if filter_mode in ['allow', 'ban']:
                    filter_info["list"] = tool_config.get('list', [])
                    
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
        await send_long_response(message, llm_response)    # Handle debug commands
    elif message.content.startswith("$ping"):
        # Simple ping command to check bot responsiveness
        await message.channel.send("Pong! 🏓")

    # Process commands (required for slash commands to work alongside message events)
    await bot.process_commands(message)


# Slash commands
@bot.slash_command(name="tools", description="Show tool information")
async def tools_slash(
    ctx, 
    mode: str = discord.Option(
        default="current",
        description="What to show",
        choices=[
            discord.OptionChoice(name="Current configuration", value="current"),
            discord.OptionChoice(name="All available tools", value="all")
        ]
    )
):
    """Simplified tools command with Discord embed"""
    await ctx.defer()
    
    if not conversation_manager.mcp_servers:
        embed = discord.Embed(
            title="⚠️ No MCP Servers", 
            description="No MCP servers configured. Tools are not available.",
            color=discord.Color.orange()
        )
        await ctx.followup.send(embed=embed)
        return
    
    try:
        async with MultiServerMCPClient(conversation_manager.mcp_servers) as mcp_client:
            all_tools = mcp_client.get_tools()
            
            if mode == "all":
                # Show all available tools
                if all_tools:
                    embed = discord.Embed(
                        title="🛠️ All Available Tools",
                        description=f"Total: **{len(all_tools)}** tools available",
                        color=discord.Color.blue()
                    )
                    
                    tool_names = sorted([tool.name for tool in all_tools])
                    
                    # Group tools in chunks for better display
                    chunk_size = 20
                    for i in range(0, len(tool_names), chunk_size):
                        chunk = tool_names[i:i + chunk_size]
                        chunk_text = "\n".join([f"• `{tool}`" for tool in chunk])
                        
                        field_name = f"Tools {i+1}-{min(i+chunk_size, len(tool_names))}"
                        embed.add_field(name=field_name, value=chunk_text, inline=True)
                        
                        # Discord has a limit of 25 fields per embed
                        if len(embed.fields) >= 24:
                            break
                    
                    if len(tool_names) > chunk_size * 24:
                        embed.add_field(
                            name="Note", 
                            value=f"Showing first {chunk_size * 24} tools. Use `/tools current` to see filtering configuration.",
                            inline=False
                        )
                    
                    await ctx.followup.send(embed=embed)
                else:
                    embed = discord.Embed(
                        title="❌ No Tools Available",
                        description="No tools available from MCP servers",
                        color=discord.Color.red()
                    )
                    await ctx.followup.send(embed=embed)
                    
            else:  # mode == "current"
                # Show current configuration
                embed = discord.Embed(
                    title="⚙️ Current Tool Configuration",
                    description="Tool filtering settings for each MCP server",
                    color=discord.Color.green()
                )
                
                for server_name, server_config in conversation_manager.mcp_servers.items():
                    tool_config = server_config.get('tools', {})
                    
                    if tool_config:
                        filter_mode = tool_config.get('mode', 'none')
                        tool_list = tool_config.get('list', [])
                        
                        # Choose emoji based on mode
                        mode_emoji = {
                            'allow': '✅',
                            'ban': '❌', 
                            'none': '🔓'                        }.get(filter_mode, '❓')
                        
                        field_value = f"{mode_emoji} **Mode:** `{filter_mode}`\n"
                        
                        if filter_mode in ['allow', 'ban'] and tool_list:
                            list_name = "Allowed" if filter_mode == 'allow' else "Banned"
                            # Display all tools without truncation
                            if len(tool_list) <= 10:
                                # For smaller lists, show inline
                                tools_display = ', '.join([f"`{tool}`" for tool in tool_list])
                                field_value += f"**{list_name} tools ({len(tool_list)}):** {tools_display}"
                            else:
                                # For larger lists, show as bullet points
                                tools_display = '\n'.join([f"• `{tool}`" for tool in tool_list[:15]])
                                if len(tool_list) > 15:
                                    tools_display += f"\n• ... and {len(tool_list)-15} more"
                                field_value += f"**{list_name} tools ({len(tool_list)}):**\n{tools_display}"
                        elif filter_mode in ['allow', 'ban']:
                            list_name = "Allowed" if filter_mode == 'allow' else "Banned"
                            field_value += f"**{list_name} tools:** (none configured)"
                        else:
                            field_value += "**Status:** All tools allowed"
                    else:
                        field_value = "🔓 **Mode:** `none`\n**Status:** All tools allowed (no filtering)"
                    
                    embed.add_field(name=f"📡 {server_name}", value=field_value, inline=False)
                
                embed.set_footer(text="Use /set-mode, /add-tool, /remove-tool to configure filtering")
                await ctx.followup.send(embed=embed)
                    
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Error getting tool information: {e}",
            color=discord.Color.red()
        )
        await ctx.followup.send(embed=embed)

@bot.slash_command(name="config", description="Show bot configuration")
async def config_slash(ctx):
    """Show bot configuration as a Discord embed"""
    await ctx.defer()
    
    config_summary = conversation_manager.get_config_summary()
    
    embed = discord.Embed(
        title="🤖 Bot Configuration",
        description="Current bot settings and configuration",
        color=discord.Color.blue()
    )
    
    # Basic configuration
    embed.add_field(
        name="🧠 LLM Configuration",
        value=f"**Model:** `{config_summary.get('llm_class', 'Unknown')}`\n**Max Tokens:** {config_summary.get('max_tokens', 'Unknown')}",
        inline=True
    )
    
    embed.add_field(
        name="💬 Conversation Settings", 
        value=f"**Max Messages:** {config_summary.get('max_messages', 'Unknown')}\n**Time Window:** {config_summary.get('max_time_window_minutes', 'Unknown')} min",
        inline=True
    )
    
    embed.add_field(
        name="🔧 Features",
        value=f"**Token Trimming:** {'✅' if config_summary.get('has_token_trimming') else '❌'}\n**MCP Servers:** {len(config_summary.get('mcp_servers', []))}",
        inline=True
    )
    
    # MCP Servers
    if config_summary.get('mcp_servers'):
        servers_text = '\n'.join([f"• `{server}`" for server in config_summary['mcp_servers']])
        embed.add_field(
            name="📡 MCP Servers",
            value=servers_text,
            inline=False
        )
    
    # Tool filtering info
    if config_summary.get('tool_filtering'):
        filtering_text = ""
        for server, filter_info in config_summary['tool_filtering'].items():
            mode = filter_info.get('mode', 'none')
            tool_count = len(filter_info.get('list', []))
            mode_emoji = {'allow': '✅', 'ban': '❌', 'none': '🔓'}.get(mode, '❓')
            filtering_text += f"{mode_emoji} **{server}:** `{mode}` ({tool_count} tools)\n"
        
        embed.add_field(
            name="🛠️ Tool Filtering",
            value=filtering_text,
            inline=False
        )
    
    # System prompt preview
    system_prompt = config_summary.get('system_prompt', '')
    if system_prompt:
        prompt_preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
        embed.add_field(
            name="📝 System Prompt",
            value=f"```{prompt_preview}```",
            inline=False
        )
    
    embed.set_footer(text="Use /tools to see detailed tool configuration")
    await ctx.followup.send(embed=embed)

@bot.slash_command(name="hello", description="Say hello!")
async def hello_slash(ctx):
    """Simple hello command as a slash command"""
    await ctx.respond("Hello! 👋")


# Configuration management functions
def save_config(config_data, file_path="configuration.yml"):
    """Save configuration back to YAML file"""
    with open(file_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False, indent=2)


async def get_available_tools():
    """Get all available tools from MCP servers"""
    if not conversation_manager.mcp_servers:
        return []
    
    try:
        async with MultiServerMCPClient(conversation_manager.mcp_servers) as mcp_client:
            return [tool.name for tool in mcp_client.get_tools()]
    except Exception as e:
        print(f"Error getting tools: {e}")
        return []


def reload_conversation_manager():
    """Reload the conversation manager with updated config"""
    global conversation_manager, config
    # Reload config from file
    config = load_config("configuration.yml")
    conversation_manager.mcp_servers = config.get('mcpServers', {})


# Tool management slash commands - simplified
@bot.slash_command(name="set-mode", description="Set tool filtering mode and update the list")
async def set_mode(
    ctx, 
    server: str = discord.Option(description="MCP server name"),
    mode: str = discord.Option(
        description="Filtering mode", 
        choices=[
            discord.OptionChoice(name="Allow only listed tools", value="allow"),
            discord.OptionChoice(name="Ban listed tools", value="ban"),
            discord.OptionChoice(name="No filtering (allow all)", value="none")
        ]
    )
):
    """Set the filtering mode for an MCP server"""
    await ctx.defer()
    
    # Check if server exists
    if server not in config.get('mcpServers', {}):
        available_servers = list(config.get('mcpServers', {}).keys())
        embed = discord.Embed(
            title="❌ Server Not Found",
            description=f"Server `{server}` not found.",
            color=discord.Color.red()
        )
        if available_servers:
            embed.add_field(
                name="Available Servers",
                value=", ".join([f"`{s}`" for s in available_servers]),
                inline=False
            )
        await ctx.followup.send(embed=embed)
        return
    
    # Update configuration
    if 'tools' not in config['mcpServers'][server]:
        config['mcpServers'][server]['tools'] = {}
    
    # Get current list or create empty one
    current_list = config['mcpServers'][server]['tools'].get('list', [])
    
    config['mcpServers'][server]['tools']['mode'] = mode
    config['mcpServers'][server]['tools']['list'] = current_list
    
    # Save configuration
    try:
        save_config(config)
        reload_conversation_manager()
        
        # Create success embed
        mode_emoji = {'allow': '✅', 'ban': '❌', 'none': '🔓'}.get(mode, '❓')
        embed = discord.Embed(
            title=f"{mode_emoji} Mode Updated",
            description=f"Successfully updated filtering mode for `{server}`",
            color=discord.Color.green()        )
        
        embed.add_field(name="Server", value=f"`{server}`", inline=True)
        embed.add_field(name="New Mode", value=f"`{mode}`", inline=True)
        
        if mode == 'none':
            embed.add_field(name="Status", value="All tools allowed", inline=False)
        else:
            list_desc = "allowed" if mode == "allow" else "banned"
            if len(current_list) <= 10:
                list_summary = ", ".join([f"`{tool}`" for tool in current_list])
            else:
                list_summary = ", ".join([f"`{tool}`" for tool in current_list[:10]])
                list_summary += f" ... and {len(current_list)-10} more"
            
            if not current_list:
                list_summary = "(empty)"
            
            embed.add_field(
                name=f"{list_desc.title()} Tools ({len(current_list)})",
                value=list_summary,
                inline=False
            )
        
        await ctx.followup.send(embed=embed)
    except Exception as e:
        embed = discord.Embed(
            title="❌ Configuration Error",
            description=f"Error saving configuration: {e}",
            color=discord.Color.red()
        )
        await ctx.followup.send(embed=embed)


@bot.slash_command(name="add-tool", description="Add a tool to the filter list")
async def add_tool(
    ctx, 
    server: str = discord.Option(description="MCP server name"),
    tool_name: str = discord.Option(description="Name of the tool to add")
):
    """Add a tool to the current filter list"""
    await ctx.defer()
    
    # Check if server exists
    if server not in config.get('mcpServers', {}):
        available_servers = list(config.get('mcpServers', {}).keys())
        embed = discord.Embed(
            title="❌ Server Not Found",
            description=f"Server `{server}` not found.",
            color=discord.Color.red()
        )
        if available_servers:
            embed.add_field(
                name="Available Servers",
                value=", ".join([f"`{s}`" for s in available_servers]),
                inline=False
            )
        await ctx.followup.send(embed=embed)
        return
    
    # Check if tool exists
    available_tools = await get_available_tools()
    if tool_name not in available_tools:
        embed = discord.Embed(
            title="❌ Tool Not Found",
            description=f"Tool `{tool_name}` not found in available tools.",
            color=discord.Color.red()
        )
        if available_tools:
            tools_preview = ", ".join([f"`{tool}`" for tool in available_tools[:10]])
            if len(available_tools) > 10:
                tools_preview += f" ... +{len(available_tools)-10} more"
            embed.add_field(
                name=f"Available Tools ({len(available_tools)} total)",
                value=tools_preview,
                inline=False
            )
        await ctx.followup.send(embed=embed)
        return
    
    # Initialize tools config if needed
    if 'tools' not in config['mcpServers'][server]:
        config['mcpServers'][server]['tools'] = {'mode': 'none', 'list': []}
    if 'list' not in config['mcpServers'][server]['tools']:
        config['mcpServers'][server]['tools']['list'] = []
    
    # Add tool if not already in list
    tool_list = config['mcpServers'][server]['tools']['list']
    mode = config['mcpServers'][server]['tools'].get('mode', 'none')
    
    if tool_name not in tool_list:
        tool_list.append(tool_name)
        
        try:
            save_config(config)
            reload_conversation_manager()
            
            embed = discord.Embed(
                title="✅ Tool Added",
                description=f"Successfully added `{tool_name}` to filter list",
                color=discord.Color.green()
            )
            
            embed.add_field(name="Server", value=f"`{server}`", inline=True)
            embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
            embed.add_field(name="Current Mode", value=f"`{mode}`", inline=True)
            
            if mode == 'none':
                embed.add_field(
                    name="💡 Note",
                    value="Mode is `none` so this list isn't active yet. Use `/set-mode` to enable filtering.",
                    inline=False
                )
            else:
                list_type = "allowed" if mode == "allow" else "banned"
                embed.add_field(
                    name="Status",
                    value=f"Tool is now in the {list_type} list",
                    inline=False
                )
            
            # Show current list size
            embed.set_footer(text=f"Total tools in list: {len(tool_list)}")
            
            await ctx.followup.send(embed=embed)
        except Exception as e:
            embed = discord.Embed(
                title="❌ Configuration Error",
                description=f"Error saving configuration: {e}",
                color=discord.Color.red()
            )
            await ctx.followup.send(embed=embed)
    else:
        list_type = "allowed" if mode == "allow" else "banned" if mode == "ban" else "filter"
        embed = discord.Embed(
            title="⚠️ Tool Already Exists",
            description=f"Tool `{tool_name}` is already in the {list_type} list for `{server}`",
            color=discord.Color.orange()
        )
        embed.set_footer(text=f"Total tools in list: {len(tool_list)}")
        await ctx.followup.send(embed=embed)


@bot.slash_command(name="remove-tool", description="Remove a tool from the filter list")
async def remove_tool(
    ctx, 
    server: str = discord.Option(description="MCP server name"),
    tool_name: str = discord.Option(description="Name of the tool to remove")
):
    """Remove a tool from the current filter list"""
    await ctx.defer()
    
    # Check if server exists
    if server not in config.get('mcpServers', {}):
        available_servers = list(config.get('mcpServers', {}).keys())
        embed = discord.Embed(
            title="❌ Server Not Found",
            description=f"Server `{server}` not found.",
            color=discord.Color.red()
        )
        if available_servers:
            embed.add_field(
                name="Available Servers",
                value=", ".join([f"`{s}`" for s in available_servers]),
                inline=False
            )
        await ctx.followup.send(embed=embed)
        return
    
    # Check if tools config exists
    if 'tools' not in config['mcpServers'][server] or 'list' not in config['mcpServers'][server]['tools']:
        embed = discord.Embed(
            title="❌ No Tool List",
            description=f"No tool list found for server `{server}`",
            color=discord.Color.red()
        )
        embed.add_field(
            name="💡 Tip",
            value="Use `/add-tool` to create a tool list or `/set-mode` to configure filtering",
            inline=False
        )
        await ctx.followup.send(embed=embed)
        return
    
    # Remove tool if in list
    tool_list = config['mcpServers'][server]['tools']['list']
    mode = config['mcpServers'][server]['tools'].get('mode', 'none')
    
    if tool_name in tool_list:
        tool_list.remove(tool_name)
        
        try:
            save_config(config)
            reload_conversation_manager()
            
            embed = discord.Embed(
                title="✅ Tool Removed",
                description=f"Successfully removed `{tool_name}` from filter list",
                color=discord.Color.green()
            )
            
            embed.add_field(name="Server", value=f"`{server}`", inline=True)
            embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
            embed.add_field(name="Current Mode", value=f"`{mode}`", inline=True)
            
            list_type = "allowed" if mode == "allow" else "banned" if mode == "ban" else "filter"
            embed.add_field(
                name="Status",
                value=f"Tool removed from {list_type} list",
                inline=False
            )
            
            # Show remaining list size
            embed.set_footer(text=f"Remaining tools in list: {len(tool_list)}")
            
            await ctx.followup.send(embed=embed)
        except Exception as e:
            embed = discord.Embed(
                title="❌ Configuration Error",
                description=f"Error saving configuration: {e}",
                color=discord.Color.red()
            )
            await ctx.followup.send(embed=embed)
    else:
        embed = discord.Embed(
            title="⚠️ Tool Not Found",
            description=f"Tool `{tool_name}` is not in the tool list for `{server}`",
            color=discord.Color.orange()
        )
        
        if tool_list:
            tools_preview = ", ".join([f"`{tool}`" for tool in tool_list[:10]])
            if len(tool_list) > 10:
                tools_preview += f" ... +{len(tool_list)-10} more"
            embed.add_field(
                name=f"Current Tools in List ({len(tool_list)})",
                value=tools_preview,
                inline=False
            )
        else:
            embed.add_field(
                name="Current Status",
                value="Tool list is empty",
                inline=False
            )
        
        await ctx.followup.send(embed=embed)



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
