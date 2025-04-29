import typer
from pathlib import Path
import os
from dotenv import load_dotenv
from discord_mcp_bot.config import load_config
from discord_mcp_bot.cogs import MCPAgent
from discord_mcp_bot.bot import MCPBot
import discord

load_dotenv()

def main():
    config_path = Path.cwd() / "configuration.yml"
    config = load_config(config_path)
    
    bot = MCPBot(config)
    bot.load_extension("discord_mcp_bot.cogs.MCPAgent")

    @bot.event
    async def on_connect():
        if bot.auto_sync_commands:
            await bot.sync_commands(register_guild_commands=True, guild_ids=[675538935472062479])
        print(f"{bot.user.name} connected.")
    
    print("Hello from discord-mcp-bot!")
    bot.run(os.getenv("DISCORD_TOKEN"))
