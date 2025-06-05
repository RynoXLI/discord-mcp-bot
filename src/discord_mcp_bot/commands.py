"""
Discord commands and event handlers for MCP Bot.

This module contains all Discord slash commands, text commands, and event handlers.
"""

import logging
from typing import Optional

import discord
from discord.ext import commands
from langchain_mcp_adapters.client import MultiServerMCPClient

from .utils import send_long_response, save_config, get_available_tools, reload_conversation_manager

logger = logging.getLogger(__name__)


class BotCommands:
    """Class to organize bot commands and event handlers."""
    
    def __init__(self, bot: commands.Bot, conversation_manager, config: dict):
        self.bot = bot
        self.conversation_manager = conversation_manager
        self.config = config
        
        # Register event handlers
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_connect)
        
        # Register slash commands
        self.setup_slash_commands()
    
    def setup_slash_commands(self):
        """Set up all slash commands."""
        
        @self.bot.slash_command(name="tools", description="Show tool information")
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
            
            if not self.conversation_manager.mcp_servers:
                embed = discord.Embed(
                    title="⚠️ No MCP Servers", 
                    description="No MCP servers configured. Tools are not available.",
                    color=discord.Color.orange()
                )
                await ctx.followup.send(embed=embed)
                return
            
            try:
                async with MultiServerMCPClient(self.conversation_manager.mcp_servers) as mcp_client:
                    all_tools = mcp_client.get_tools()
                    
                    if mode == "all":
                        await self._handle_tools_all_mode(ctx, all_tools)
                    else:  # mode == "current"
                        await self._handle_tools_current_mode(ctx)
                        
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Error",
                    description=f"Error getting tool information: {e}",
                    color=discord.Color.red()
                )
                await ctx.followup.send(embed=embed)
        
        @self.bot.slash_command(name="config", description="Show bot configuration")
        async def config_slash(ctx):
            """Show bot configuration as a Discord embed"""
            await ctx.defer()
            
            config_summary = self.conversation_manager.get_config_summary()
            
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
        
        @self.bot.slash_command(name="hello", description="Say hello!")
        async def hello_slash(ctx):
            """Simple hello command as a slash command"""
            await ctx.respond("Hello! 👋")
        
        @self.bot.slash_command(name="set-mode", description="Set tool filtering mode and update the list")
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
            await self._handle_set_mode(ctx, server, mode)
        
        @self.bot.slash_command(name="add-tool", description="Add a tool to the filter list")
        async def add_tool(
            ctx, 
            server: str = discord.Option(description="MCP server name"),
            tool_name: str = discord.Option(description="Name of the tool to add")
        ):
            """Add a tool to the current filter list"""
            await ctx.defer()
            await self._handle_add_tool(ctx, server, tool_name)
        
        @self.bot.slash_command(name="remove-tool", description="Remove a tool from the filter list")
        async def remove_tool(
            ctx, 
            server: str = discord.Option(description="MCP server name"),
            tool_name: str = discord.Option(description="Name of the tool to remove")
        ):
            """Remove a tool from the current filter list"""
            await ctx.defer()
            await self._handle_remove_tool(ctx, server, tool_name)
    
    async def _handle_tools_all_mode(self, ctx, all_tools):
        """Handle the 'all' mode for tools command."""
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
    
    async def _handle_tools_current_mode(self, ctx):
        """Handle the 'current' mode for tools command."""
        embed = discord.Embed(
            title="⚙️ Current Tool Configuration",
            description="Tool filtering settings for each MCP server",
            color=discord.Color.green()
        )
        
        for server_name, server_config in self.conversation_manager.mcp_servers.items():
            tool_config = server_config.get('tools', {})
            
            if tool_config:
                filter_mode = tool_config.get('mode', 'none')
                tool_list = tool_config.get('list', [])
                
                # Choose emoji based on mode
                mode_emoji = {
                    'allow': '✅',
                    'ban': '❌', 
                    'none': '🔓'
                }.get(filter_mode, '❓')
                
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
    
    async def _handle_set_mode(self, ctx, server: str, mode: str):
        """Handle set-mode command."""
        # Check if server exists
        if server not in self.config.get('mcpServers', {}):
            available_servers = list(self.config.get('mcpServers', {}).keys())
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
        if 'tools' not in self.config['mcpServers'][server]:
            self.config['mcpServers'][server]['tools'] = {}
        
        # Get current list or create empty one
        current_list = self.config['mcpServers'][server]['tools'].get('list', [])
        
        self.config['mcpServers'][server]['tools']['mode'] = mode
        self.config['mcpServers'][server]['tools']['list'] = current_list
        
        # Save configuration
        try:
            save_config(self.config)
            reload_conversation_manager(self.conversation_manager)
            
            # Create success embed
            mode_emoji = {'allow': '✅', 'ban': '❌', 'none': '🔓'}.get(mode, '❓')
            embed = discord.Embed(
                title=f"{mode_emoji} Mode Updated",
                description=f"Successfully updated filtering mode for `{server}`",
                color=discord.Color.green()
            )
            
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
    
    async def _handle_add_tool(self, ctx, server: str, tool_name: str):
        """Handle add-tool command."""
        # Check if server exists
        if server not in self.config.get('mcpServers', {}):
            available_servers = list(self.config.get('mcpServers', {}).keys())
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
        available_tools = await get_available_tools(self.conversation_manager.mcp_servers)
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
        if 'tools' not in self.config['mcpServers'][server]:
            self.config['mcpServers'][server]['tools'] = {'mode': 'none', 'list': []}
        if 'list' not in self.config['mcpServers'][server]['tools']:
            self.config['mcpServers'][server]['tools']['list'] = []
        
        # Add tool if not already in list
        tool_list = self.config['mcpServers'][server]['tools']['list']
        mode = self.config['mcpServers'][server]['tools'].get('mode', 'none')
        
        if tool_name not in tool_list:
            tool_list.append(tool_name)
            
            try:
                save_config(self.config)
                reload_conversation_manager(self.conversation_manager)
                
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
    
    async def _handle_remove_tool(self, ctx, server: str, tool_name: str):
        """Handle remove-tool command."""
        # Check if server exists
        if server not in self.config.get('mcpServers', {}):
            available_servers = list(self.config.get('mcpServers', {}).keys())
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
        if 'tools' not in self.config['mcpServers'][server] or 'list' not in self.config['mcpServers'][server]['tools']:
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
        tool_list = self.config['mcpServers'][server]['tools']['list']
        mode = self.config['mcpServers'][server]['tools'].get('mode', 'none')
        
        if tool_name in tool_list:
            tool_list.remove(tool_name)
            
            try:
                save_config(self.config)
                reload_conversation_manager(self.conversation_manager)
                
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
    
    # Event handlers
    async def on_ready(self):
        """Called when the bot is ready."""
        print(f"We have logged in as {self.bot.user}")
    
    async def on_connect(self):
        """Called when the bot connects."""
        if self.bot.auto_sync_commands:
            await self.bot.sync_commands(register_guild_commands=True, guild_ids=[675538935472062479])
        print(f"{self.bot.user.name} connected.")
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        if message.author == self.bot.user:
            return

        # Check if this message is part of a reply chain
        if message.reference and message.reference.message_id:
            try:
                replied_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                if replied_message.author == self.bot.user:
                    # Get conversation formatted for LLM using replies mode
                    formatted_messages = await self.conversation_manager.get_reply_conversation(
                        message
                    )

                    # Show typing indicator while getting LLM response
                    async with message.channel.typing():
                        # Get LLM response
                        llm_response = await self.conversation_manager.get_llm_response(
                            formatted_messages
                        )
                    
                    # Send response (potentially split into multiple messages)
                    await send_long_response(message, llm_response)

                    logger.info(f"LLM Response: {llm_response}")
            except (discord.NotFound, discord.Forbidden):
                pass

        # Check if bot is mentioned, but don't double-reply if it's already handled in reply chain
        elif self.bot.user.mentioned_in(message):
            # Get conversation formatted for LLM using channel mode for recent context
            formatted_messages = await self.conversation_manager.get_channel_conversation(
                message
            )
            
            logger.info(f"Channel conversation context: {len(formatted_messages)} messages")
            
            # Show typing indicator while getting LLM response
            async with message.channel.typing():
                # Get LLM response
                llm_response = await self.conversation_manager.get_llm_response(
                    formatted_messages
                )

            # Send response (potentially split into multiple messages)
            await send_long_response(message, llm_response)
        
        # Handle debug commands
        elif message.content.startswith("$ping"):
            # Simple ping command to check bot responsiveness
            await message.channel.send("Pong! 🏓")

        # Process commands (required for slash commands to work alongside message events)
        await self.bot.process_commands(message)
