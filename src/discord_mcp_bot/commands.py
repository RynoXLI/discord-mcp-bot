"""
Discord commands and event handlers for MCP Bot.

This module contains all Discord slash commands, text commands, and event handlers.
"""

import logging
from typing import Optional, Tuple # Added Optional, Tuple

import discord
from discord.ext import commands
from langchain_mcp_adapters.client import MultiServerMCPClient
from .conversation import ConversationManager

from .utils import (
    send_long_response,
    save_config,
    get_available_tools,
    reload_conversation_manager,
)

logger = logging.getLogger(__name__)


class ThoughtsView(discord.ui.View):
    def __init__(self, main_response: str, thoughts_content: str, original_author: discord.User, original_message_id: int):
        super().__init__(timeout=300.0)  # 5 minutes timeout
        self.main_response = main_response
        self.thoughts_content = thoughts_content
        self.original_author = original_author
        self.thoughts_visible = False
        
        # Create and add the button
        self.toggle_thoughts_button = discord.ui.Button(
            label="Show Thoughts",
            style=discord.ButtonStyle.secondary,
            custom_id=f"toggle_thoughts_{original_message_id}" 
        )
        self.toggle_thoughts_button.callback = self.toggle_thoughts_callback
        self.add_item(self.toggle_thoughts_button)

    def create_embed(self) -> discord.Embed:
        embed = discord.Embed(
            description=self.main_response,
            color=discord.Color.blue() 
        )
        if self.thoughts_visible:
            # Ensure thoughts are not too long for an embed field
            display_thoughts = self.thoughts_content
            if len(display_thoughts) > 1000: # Embed field value limit is 1024
                display_thoughts = display_thoughts[:1000] + "..."
            embed.add_field(name="🤔 My Thought Process", value=f"```{display_thoughts}```", inline=False)
        return embed

    async def toggle_thoughts_callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.original_author.id:
            await interaction.response.send_message("Only the user who triggered the bot can toggle thoughts.", ephemeral=True)
            return

        self.thoughts_visible = not self.thoughts_visible
        
        # Update button properties
        self.toggle_thoughts_button.label = "Hide Thoughts" if self.thoughts_visible else "Show Thoughts"
        self.toggle_thoughts_button.style = discord.ButtonStyle.primary if self.thoughts_visible else discord.ButtonStyle.secondary
        
        new_embed = self.create_embed()
        try:
            await interaction.response.edit_message(embed=new_embed, view=self)
        except discord.errors.InteractionResponded:
            # If the interaction was already responded to (e.g. by a quick follow-up ephemeral message)
            # try to edit the original message directly if possible.
            # This might happen if the ephemeral message above is sent and then we try to edit.
            # For simplicity, we'll assume the primary path is edit_message.
            # Handling complex interaction state is beyond this immediate scope.
            logger.warning(f"Interaction {interaction.id} already responded, cannot edit message for thoughts toggle.")
            # As a fallback, could try: await interaction.edit_original_response(embed=new_embed, view=self)
            # but this also might fail if the initial response was ephemeral.

class BotCommands:
    """Class to organize bot commands and event handlers."""

    def __init__(self, bot: commands.Bot, conversation_manager: ConversationManager, config: dict):
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
                    discord.OptionChoice(name="All available tools", value="all"),
                ],
            ),
        ):
            """Simplified tools command with Discord embed"""
            await ctx.defer()

            if not self.conversation_manager.mcp_servers:
                embed = discord.Embed(
                    title="⚠️ No MCP Servers",
                    description="No MCP servers configured. Tools are not available.",                    color=discord.Color.orange(),
                )
                await ctx.followup.send(embed=embed)
                return

            try:
                async with MultiServerMCPClient(
                    self.conversation_manager.mcp_servers
                ) as mcp_client:
                    all_tools = mcp_client.get_tools()

                    if mode == "all":
                        await self._handle_tools_all_mode(ctx, all_tools, mcp_client)
                    else:  # mode == "current"
                        await self._handle_tools_current_mode(ctx)

            except Exception as e:
                embed = discord.Embed(
                    title="❌ Error",
                    description=f"Error getting tool information: {e}",
                    color=discord.Color.red(),
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
                color=discord.Color.blue(),
            )

            # Basic configuration
            embed.add_field(
                name="🧠 LLM Configuration",
                value=f"**Model:** `{config_summary.get('llm_class', 'Unknown')}`\n**Max Tokens:** {config_summary.get('max_tokens', 'Unknown')}",
                inline=True,
            )

            embed.add_field(
                name="💬 Conversation Settings",
                value=f"**Max Messages:** {config_summary.get('max_messages', 'Unknown')}\n**Time Window:** {config_summary.get('max_time_window_minutes', 'Unknown')} min",
                inline=True,
            )

            embed.add_field(
                name="🔧 Features",
                value=f"**Token Trimming:** {'✅' if config_summary.get('has_token_trimming') else '❌'}\n**MCP Servers:** {len(config_summary.get('mcp_servers', []))}",
                inline=True,
            )

            # MCP Servers
            if config_summary.get("mcp_servers"):
                servers_text = "\n".join(
                    [f"• `{server}`" for server in config_summary["mcp_servers"]]
                )
                embed.add_field(name="📡 MCP Servers", value=servers_text, inline=False)

            # Tool filtering info
            if config_summary.get("tool_filtering"):
                filtering_text = ""
                for server, filter_info in config_summary["tool_filtering"].items():
                    mode = filter_info.get("mode", "none")
                    tool_count = len(filter_info.get("list", []))
                    mode_emoji = {"allow": "✅", "ban": "❌", "none": "🔓"}.get(
                        mode, "❓"
                    )
                    filtering_text += (
                        f"{mode_emoji} **{server}:** `{mode}` ({tool_count} tools)\n"
                    )

                embed.add_field(
                    name="🛠️ Tool Filtering", value=filtering_text, inline=False
                )

            # System prompt preview
            system_prompt = config_summary.get("system_prompt", "")
            if system_prompt:
                prompt_preview = (
                    system_prompt[:100] + "..."
                    if len(system_prompt) > 100
                    else system_prompt
                )
                
                embed.add_field(
                    name="📝 System Prompt",
                    value=f"```{prompt_preview}```",
                    inline=False,
                )
                
            embed.set_footer(text="Use /tools to see detailed tool configuration")
            await ctx.followup.send(embed=embed)

        @self.bot.slash_command(name="hello", description="Say hello!")
        async def hello_slash(ctx):
            """Simple hello command as a slash command"""
            await ctx.respond("Hello! 👋")

        @self.bot.slash_command(name="new", description="Start a fresh conversation without previous history")
        async def new_slash(ctx):
            """Start a fresh conversation without looking at previous history"""
            embed = discord.Embed(
                title="🆕 Fresh Chat Started",
                description="I'm ready for a new conversation! I won't reference any previous messages in this channel. What would you like to talk about?",
                color=discord.Color.green(),
            )
            embed.set_footer(text="Reply to this message or mention me to continue the fresh conversation")
            await ctx.respond(embed=embed)

        @self.bot.slash_command(
            name="set-mode", description="Set tool filtering mode and update the list"
        )
        async def set_mode(
            ctx,
            server: str = discord.Option(description="MCP server name"),
            mode: str = discord.Option(
                description="Filtering mode",
                choices=[
                    discord.OptionChoice(name="Allow only listed tools", value="allow"),
                    discord.OptionChoice(name="Ban listed tools", value="ban"),
                    discord.OptionChoice(name="No filtering (allow all)", value="none"),
                ],
            ),
        ):
            """Set the filtering mode for an MCP server"""
            await ctx.defer()
            await self._handle_set_mode(ctx, server, mode)

        @self.bot.slash_command(
            name="add-tool", description="Add a tool to the filter list"
        )
        async def add_tool(
            ctx,
            server: str = discord.Option(description="MCP server name"),
            tool_name: str = discord.Option(description="Name of the tool to add"),
        ):
            """Add a tool to the current filter list"""
            await ctx.defer()
            await self._handle_add_tool(ctx, server, tool_name)

        @self.bot.slash_command(
            name="remove-tool", description="Remove a tool from the filter list"
        )
        async def remove_tool(
            ctx,
            server: str = discord.Option(description="MCP server name"),
            tool_name: str = discord.Option(description="Name of the tool to remove"),
        ):
            """Remove a tool from the current filter list"""
            await ctx.defer()
            await self._handle_remove_tool(ctx, server, tool_name)

    async def _handle_tools_all_mode(self, ctx, all_tools, mcp_client=None):
        """Handle the 'all' mode for tools command."""
        if all_tools:
            embed = discord.Embed(
                title="🛠️ All Available Tools",
                description=f"Total: **{len(all_tools)}** tools available",
                color=discord.Color.blue(),
            )

            # Group tools by server
            tools_by_server = {}
            
            # If we have the mcp_client, we can get tools by server
            if mcp_client and hasattr(mcp_client, 'servers'):
                for server_name in self.conversation_manager.mcp_servers.keys():
                    try:
                        server_tools = [tool for tool in all_tools if tool.name.startswith(f"{server_name}_")]
                        if server_tools:
                            tools_by_server[server_name] = sorted([tool.name.replace(f"{server_name}_", "", 1) for tool in server_tools])
                    except Exception:
                        # Fallback: group all tools under their server names if available
                        pass
            
            # If we couldn't group by server, show all tools
            if not tools_by_server:
                tool_names = sorted([tool.name for tool in all_tools])
                tools_by_server["All Servers"] = tool_names

            # Display tools grouped by server
            for server_name, tool_names in tools_by_server.items():
                if not tool_names:
                    continue
                    
                # Group tools in chunks for better display
                chunk_size = 15
                if len(tool_names) <= chunk_size:
                    # For smaller lists, show inline
                    tools_display = "\n".join([f"• `{tool}`" for tool in tool_names])
                    embed.add_field(
                        name=f"📡 {server_name} ({len(tool_names)} tools)",
                        value=tools_display,
                        inline=False
                    )
                else:
                    # For larger lists, show as bullet points with truncation
                    first_chunk = tool_names[:chunk_size]
                    tools_display = "\n".join([f"• `{tool}`" for tool in first_chunk])
                    
                    if len(tool_names) > chunk_size:
                        remaining = len(tool_names) - chunk_size
                        tools_display += f"\n\n*...and {remaining} more tools*"
                    
                    embed.add_field(
                        name=f"📡 {server_name} ({len(tool_names)} tools)",
                        value=tools_display,
                        inline=False
                    )

            await ctx.followup.send(embed=embed)
        else:
            embed = discord.Embed(
                title="❌ No Tools Available",
                description="No tools available from MCP servers",
                color=discord.Color.red(),
            )
            await ctx.followup.send(embed=embed)

    async def _handle_tools_current_mode(self, ctx):
        """Handle the 'current' mode for tools command."""
        embed = discord.Embed(
            title="⚙️ Current Tool Configuration",
            description="Tool filtering settings for each MCP server",
            color=discord.Color.green(),
        )

        for server_name, server_config in self.conversation_manager.mcp_servers.items():
            tool_config = server_config.get("tools", {})

            if tool_config:
                filter_mode = tool_config.get("mode", "none")
                tool_list = tool_config.get("list", [])                # Choose emoji based on mode
                mode_emoji = {"allow": "✅", "ban": "❌", "none": "🔓"}.get(
                    filter_mode, "❓"
                )

                field_value = f"{mode_emoji} **Mode:** `{filter_mode}`\n"
                
                if filter_mode in ["allow", "ban"] and tool_list:
                    list_name = "Allowed" if filter_mode == "allow" else "Banned"
                    # Display all tools without truncation
                    if len(tool_list) <= 10:
                        # For smaller lists, show inline
                        tools_display = ", ".join([f"`{tool}`" for tool in tool_list])
                        field_value += (
                            f"**{list_name} tools ({len(tool_list)}):** {tools_display}"
                        )
                    else:
                        # For larger lists, show as bullet points
                        tools_display = "\n".join(
                            [f"• `{tool}`" for tool in tool_list[:15]]
                        )
                        if len(tool_list) > 15:
                            tools_display += f"\n\n*...and {len(tool_list) - 15} more tools*"
                        field_value += f"**{list_name} tools ({len(tool_list)}):**\n{tools_display}"
                elif filter_mode in ["allow", "ban"]:
                    list_name = "Allowed" if filter_mode == "allow" else "Banned"
                    field_value += f"**{list_name} tools:** (none configured)"
                else:
                    field_value += "**Status:** All tools allowed"
            else:
                field_value = (
                    "🔓 **Mode:** `none`\n**Status:** All tools allowed (no filtering)"
                )

            embed.add_field(name=f"📡 {server_name}", value=field_value, inline=False)

        embed.set_footer(
            text="Use /set-mode, /add-tool, /remove-tool to configure filtering"
        )
        await ctx.followup.send(embed=embed)

    async def _handle_set_mode(self, ctx, server: str, mode: str):
        """Handle set-mode command."""
        # Check if server exists
        if server not in self.config.get("mcpServers", {}):
            available_servers = list(self.config.get("mcpServers", {}).keys())
            embed = discord.Embed(
                title="❌ Server Not Found",
                description=f"Server `{server}` not found.",
                color=discord.Color.red(),
            )
            if available_servers:
                embed.add_field(
                    name="Available Servers",
                    value=", ".join([f"`{s}`" for s in available_servers]),
                    inline=False,
                )
            await ctx.followup.send(embed=embed)
            return

        # Update configuration
        if "tools" not in self.config["mcpServers"][server]:
            self.config["mcpServers"][server]["tools"] = {}

        # Get current list or create empty one
        current_list = self.config["mcpServers"][server]["tools"].get("list", [])

        self.config["mcpServers"][server]["tools"]["mode"] = mode
        self.config["mcpServers"][server]["tools"]["list"] = current_list

        # Save configuration
        try:
            save_config(self.config)
            reload_conversation_manager(self.conversation_manager)

            # Create success embed
            mode_emoji = {"allow": "✅", "ban": "❌", "none": "🔓"}.get(mode, "❓")
            embed = discord.Embed(
                title=f"{mode_emoji} Mode Updated",
                description=f"Successfully updated filtering mode for `{server}`",
                color=discord.Color.green(),
            )

            embed.add_field(name="Server", value=f"`{server}`", inline=True)
            embed.add_field(name="New Mode", value=f"`{mode}`", inline=True)            
            if mode == "none":
                embed.add_field(name="Status", value="All tools allowed", inline=False)
            else:
                list_desc = "allowed" if mode == "allow" else "banned"
                
                if len(current_list) <= 10:
                    list_summary = ", ".join([f"`{tool}`" for tool in current_list])
                else:
                    # For larger lists, show as bullet points for better readability
                    sorted_list = sorted(current_list)
                    list_summary = "\n".join([f"• `{tool}`" for tool in sorted_list[:15]])
                    
                    if len(sorted_list) > 15:
                        list_summary += f"\n\n*...and {len(sorted_list) - 15} more tools*"

                if not current_list:
                    list_summary = "(empty)"

                embed.add_field(
                    name=f"{list_desc.title()} Tools ({len(current_list)})",
                    value=list_summary,
                    inline=False,
                )

            await ctx.followup.send(embed=embed)
        except Exception as e:
            embed = discord.Embed(
                title="❌ Configuration Error",
                description=f"Error saving configuration: {e}",
                color=discord.Color.red(),
            )
            await ctx.followup.send(embed=embed)

    async def _handle_add_tool(self, ctx, server: str, tool_name: str):
        """Handle add-tool command."""
        # Check if server exists
        if server not in self.config.get("mcpServers", {}):
            available_servers = list(self.config.get("mcpServers", {}).keys())
            embed = discord.Embed(
                title="❌ Server Not Found",
                description=f"Server `{server}` not found.",
                color=discord.Color.red(),
            )
            if available_servers:
                embed.add_field(
                    name="Available Servers",
                    value=", ".join([f"`{s}`" for s in available_servers]),
                    inline=False,
                )
            await ctx.followup.send(embed=embed)
            return        # Check if tool exists
        available_tools = await get_available_tools(
            self.conversation_manager.mcp_servers
        )
        if tool_name not in available_tools:
            embed = discord.Embed(
                title="❌ Tool Not Found",
                description=f"Tool `{tool_name}` not found in available tools.",
                color=discord.Color.red(),
            )
            if available_tools:
                sorted_tools = sorted(available_tools)
                
                # Calculate how many pages of tools to display
                total_tools = len(sorted_tools)
                tools_per_page = 15
                total_pages = (total_tools + tools_per_page - 1) // tools_per_page  # Ceiling division
                
                # Show the first page of tools as bullet points
                first_page_tools = sorted_tools[:tools_per_page]
                tools_preview = "\n".join([f"• `{tool}`" for tool in first_page_tools])
                
                if total_tools > tools_per_page:
                    # Add a note about additional tools
                    remaining_tools = total_tools - tools_per_page
                    embed.add_field(
                        name=f"Available Tools (Page 1/{total_pages}, {total_tools} total)",
                        value=f"{tools_preview}\n\n*Use `/tools all` to see all {total_tools} available tools*",
                        inline=False,
                    )
                else:
                    embed.add_field(
                        name=f"Available Tools ({total_tools} total)",
                        value=tools_preview,
                        inline=False,
                    )
            await ctx.followup.send(embed=embed)
            return

        # Initialize tools config if needed
        if "tools" not in self.config["mcpServers"][server]:
            self.config["mcpServers"][server]["tools"] = {"mode": "none", "list": []}
        if "list" not in self.config["mcpServers"][server]["tools"]:
            self.config["mcpServers"][server]["tools"]["list"] = []

        # Add tool if not already in list
        tool_list = self.config["mcpServers"][server]["tools"]["list"]
        mode = self.config["mcpServers"][server]["tools"].get("mode", "none")

        if tool_name not in tool_list:
            tool_list.append(tool_name)

            try:
                save_config(self.config)
                reload_conversation_manager(self.conversation_manager)

                embed = discord.Embed(
                    title="✅ Tool Added",
                    description=f"Successfully added `{tool_name}` to filter list",
                    color=discord.Color.green(),
                )

                embed.add_field(name="Server", value=f"`{server}`", inline=True)
                embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
                embed.add_field(name="Current Mode", value=f"`{mode}`", inline=True)

                if mode == "none":
                    embed.add_field(
                        name="💡 Note",
                        value="Mode is `none` so this list isn't active yet. Use `/set-mode` to enable filtering.",
                        inline=False,
                    )
                else:
                    list_type = "allowed" if mode == "allow" else "banned"
                    embed.add_field(
                        name="Status",
                        value=f"Tool is now in the {list_type} list",
                        inline=False,
                    )

                # Show current list size
                embed.set_footer(text=f"Total tools in list: {len(tool_list)}")

                await ctx.followup.send(embed=embed)
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Configuration Error",
                    description=f"Error saving configuration: {e}",
                    color=discord.Color.red(),
                )
                await ctx.followup.send(embed=embed)
        else:
            list_type = (
                "allowed"
                if mode == "allow"
                else "banned"
                if mode == "ban"
                else "filter"
            )
            embed = discord.Embed(
                title="⚠️ Tool Already Exists",
                description=f"Tool `{tool_name}` is already in the {list_type} list for `{server}`",
                color=discord.Color.orange(),
            )
            embed.set_footer(text=f"Total tools in list: {len(tool_list)}")
            await ctx.followup.send(embed=embed)

    async def _handle_remove_tool(self, ctx, server, tool_name):
        """Remove a tool from the current filter list"""
        # Check if server exists
        if server not in self.config.get("mcpServers", {}):
            available_servers = list(self.config.get("mcpServers", {}).keys())
            embed = discord.Embed(
                title="❌ Server Not Found",
                description=f"Server `{server}` not found.",
                color=discord.Color.red(),
            )
            if available_servers:
                embed.add_field(
                    name="Available Servers",
                    value=", ".join([f"`{s}`" for s in available_servers]),
                    inline=False,
                )
            await ctx.followup.send(embed=embed)
            return

        # Check if tools config exists
        if (
            "tools" not in self.config["mcpServers"][server]
            or "list" not in self.config["mcpServers"][server]["tools"]
        ):
            embed = discord.Embed(
                title="❌ No Tool List",
                description=f"No tool list found for server `{server}`",
                color=discord.Color.red(),
            )
            embed.add_field(
                name="💡 Tip",
                value="Use `/add-tool` to create a tool list or `/set-mode` to configure filtering",
                inline=False,
            )
            await ctx.followup.send(embed=embed)
            return

        # Remove tool if in list
        tool_list = self.config["mcpServers"][server]["tools"]["list"]
        mode = self.config["mcpServers"][server]["tools"].get("mode", "none")

        if tool_name in tool_list:
            tool_list.remove(tool_name)

            try:
                save_config(self.config)
                reload_conversation_manager(self.conversation_manager)

                embed = discord.Embed(
                    title="✅ Tool Removed",
                    description=f"Successfully removed `{tool_name}` from filter list",
                    color=discord.Color.green(),
                )

                embed.add_field(name="Server", value=f"`{server}`", inline=True)
                embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
                embed.add_field(name="Current Mode", value=f"`{mode}`", inline=True)

                list_type = (
                    "allowed"
                    if mode == "allow"
                    else "banned"
                    if mode == "ban"
                    else "filter"
                )
                embed.add_field(
                    name="Status",
                    value=f"Tool removed from {list_type} list",
                    inline=False,
                )

                # Show remaining list size
                embed.set_footer(text=f"Remaining tools in list: {len(tool_list)}")

                await ctx.followup.send(embed=embed)
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Configuration Error",
                    description=f"Error saving configuration: {e}",
                    color=discord.Color.red(),                )
                await ctx.followup.send(embed=embed)
        else:
            embed = discord.Embed(
                title="⚠️ Tool Not Found",
                description=f"Tool `{tool_name}` is not in the tool list for `{server}`",
                color=discord.Color.orange(),
            )

            if tool_list:
                sorted_tools = sorted(tool_list)
                
                # Calculate how many tools to display
                total_tools = len(sorted_tools)
                tools_per_page = 15
                
                # Show the tools as bullet points
                first_page_tools = sorted_tools[:tools_per_page]
                tools_preview = "\n".join([f"• `{tool}`" for tool in first_page_tools])
                
                if total_tools > tools_per_page:
                    # Add a note about additional tools
                    remaining_tools = total_tools - tools_per_page
                    tools_preview += f"\n\n*...and {remaining_tools} more tools*"
                
                embed.add_field(
                    name=f"Current Tools in List ({total_tools})",
                    value=tools_preview,
                    inline=False,
                )
            else:
                embed.add_field(
                    name="Current Status", value="Tool list is empty", inline=False
                )

            await ctx.followup.send(embed=embed)

    async def create_response_with_thoughts(
        self,
        message: discord.Message,
        main_response: str,
        thoughts: Optional[str]
    ) -> Tuple[discord.Embed, Optional[ThoughtsView]]:
        """Creates an embed with a button to toggle thoughts, if thoughts are present."""
        if not thoughts:
            embed = discord.Embed(description=main_response, color=discord.Color.blue())
            return embed, None

        view = ThoughtsView(
            main_response=main_response,
            thoughts_content=thoughts,
            original_author=message.author,
            original_message_id=message.id
        )
        
        embed = view.create_embed() # Initial embed state
        return embed, view

    # Event handlers
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.bot.user.name} (ID: {self.bot.user.id})")
        logger.info("------")
        # Set presence
        await self.bot.change_presence(
            activity=discord.Game(name=self.config.get("status_message", "/help"))
        )
        kwargs = self.config.get("system_prompt", {}).get("args", {})
        kwargs.pop("bot_username", None)  # Remove bot_username if present
        kwargs.pop("bot_userid", None)  # Remove bot_userid if present
        self.conversation_manager.system_prompt = self.conversation_manager.system_prompt.format(bot_username=self.bot.user.name, bot_userid=self.bot.user.id, **kwargs)
        logger.info("Bot system prompt set:")
        logger.info(self.conversation_manager.system_prompt)  # Print the system prompt for debugging

        # temp, but run once to get tools
        async with self.conversation_manager.get_agent() as agent:
            pass
    
    async def on_connect(self):
        """Called when the bot connects."""
        logger.info("Bot connected to Discord.")

    async def on_message(self, message: discord.Message):
        """Event handler for incoming messages."""        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return
    
        # Process messages mentioning the bot or direct messages
        if (
            self.bot.user.mentioned_in(message)
            or isinstance(message.channel, discord.DMChannel)
        ):
            async with message.channel.typing():
                try:
                    # If the message is a reply, we can use the reference to get context
                    if message.reference:
                        history = await self.conversation_manager.get_reply_chain(message)
                    else:
                        history = await self.conversation_manager.get_recent_channel_messages(message)

                    # Log the message history
                    logger.info(f"Received request from {message.author.name} (ID: {message.author.id}) in {message.channel}")
                    logger.info(f"Message history ({len(history)} messages):")
                    for i, msg in enumerate(history):
                        author = msg.author.name
                        content = msg.content if len(msg.content) < 100 else f"{msg.content[:97]}..."
                        logger.info(f"  [{i+1}] {author}: {content}")
                    
                    formatted_messages = (
                        self.conversation_manager.format_messages_for_llm(history)
                    )

                    # Get LLM response
                    llm_response, thoughts = await self.conversation_manager.get_llm_response(formatted_messages)
                    
                    if thoughts:
                        embed, view = await self.create_response_with_thoughts(message, llm_response, thoughts)
                        await message.reply(embed=embed, view=view)
                    else:
                        await send_long_response(message, llm_response)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    error_embed = discord.Embed(
                        title="❌ Error",
                        description="I'm sorry, I encountered an error while trying to respond.",
                        color=discord.Color.red(),
                    )
                    try:
                        await message.reply(embed=error_embed)
                    except discord.HTTPException as http_e:
                        logger.error(f"Failed to send error reply: {http_e}")

        # Allow processing of other commands (e.g., text commands if any)
        # await self.bot.process_commands(message)


class ConfigCog(commands.Cog):
    """Cog for configuration commands."""

    def __init__(self, bot, conversation_manager, config):
        self.bot = bot
        self.conversation_manager = conversation_manager
        self.config = config

    @commands.command(name="reload-config", help="Reloads the bot configuration.")
    @commands.is_owner()
    async def reload_config(self, ctx):
        """Reloads the bot configuration from configuration.yml."""
        try:
            # Re-initialize the conversation manager with the new config
            self.conversation_manager = await reload_conversation_manager(self.config)
            # Update the BotCommands instance's conversation_manager as well
            # This assumes BotCommands is accessible, e.g., via bot.cogs or a shared reference
            for cog_name, cog_instance in self.bot.cogs.items():
                if hasattr(cog_instance, "conversation_manager"):
                    cog_instance.conversation_manager = self.conversation_manager
            # Also update the main BotCommands instance if it's not a cog (might need a direct reference)
            # For this example, we assume it's handled if BotCommands is part of a cog or globally accessible

            await ctx.send("Configuration reloaded successfully.")
            logger.info("Configuration reloaded by command.")
        except Exception as e:
            await ctx.send(f"Error reloading configuration: {e}")
            logger.error(f"Error reloading configuration by command: {e}")

    @commands.command(name="reset-conversation", help="Resets your conversation history.")
    async def reset_conversation(self, ctx):
        """Resets the conversation history for the invoking user."""
        await self.conversation_manager.reset_conversation(ctx.author.id)
        await ctx.send(
            "Your conversation history has been reset. I'll start fresh with you!"
        )

    @commands.command(name="hello", help="Say hello!")
    async def hello(self, ctx):
        """Simple hello command"""
        await ctx.send("Hello! 👋")


async def setup(bot: commands.Bot, conversation_manager, config: dict):
    """Setup function to add cogs to the bot."""
    # Add BotCommands (which includes slash commands and event handlers)
    # Note: If BotCommands is not a Cog, it needs to be instantiated and event handlers registered directly.
    # For simplicity, assuming it's handled or BotCommands methods are part of a Cog.
    # If BotCommands is meant to be the primary command handler and not a cog:
    bot_commands_instance = BotCommands(bot, conversation_manager, config)
    # Events and slash commands are registered in BotCommands.__init__

    # Add ConfigCog
    await bot.add_cog(ConfigCog(bot, conversation_manager, config))
    logger.info("BotCommands and ConfigCog loaded.")
