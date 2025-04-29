import discord
from discord.ext import commands
from discord_mcp_bot.llm import get_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class MCPAgent(commands.Cog): # create a class for our cog that inherits from commands.Cog
    # this class is used to create a cog, which is a module that can be added to the bot

    def __init__(self, bot): # this is a special method that is called when the cog is loaded
        self.bot = bot
        self.config = bot.config

    @discord.slash_command(name="hello", description="Say hello to the bot")
    async def hello(self, ctx: discord.ApplicationContext):
        await ctx.respond("Hey!")
        await ctx.respond("Pong!")

def setup(bot):
    """
    Setup function to add the cog to the bot.
    
    Args:
        bot (discord.Bot): The Discord bot instance.
    """
    config = bot.config

    # dynamically create slash commands for each LLM
    for agent, agent_config in config["agents"].items():

        agent = get_agent(agent_config, config["llms"], config["mcpServers"])
        cmd = agent_config["command"]
        
        @discord.slash_command(name=cmd, description="Ask the bot a question", cog=MCPAgent)
        async def func(ctx: discord.ApplicationContext, question: str):
            async for event in agent.astream(
                {
                    "messages": [{
                        "role": "user", "content": question
                    }]
                },
                stream_mode="values"
            ):
                if "messages" in event:
                    if isinstance(event['messages'][-1], AIMessage):
                        await ctx.respond(event['messages'][-1].text())
            # await ctx.respond(ai_msg.text())
        print(f"Creating slash command for {agent}")
        setattr(MCPAgent, cmd, func)
        bot.add_application_command(getattr(MCPAgent, cmd))
    # Add the cog to the bot
    cog = MCPAgent(bot)
    
    bot.add_cog(cog)