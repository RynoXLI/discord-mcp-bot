import discord
from discord.ext import commands
from discord_mcp_bot.llm import get_agent, get_thread_name
from langchain_core.messages import AIMessage

class Agent(commands.Cog): # create a class for our cog that inherits from commands.Cog
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

        cmd = agent_config["command"]

        @discord.slash_command(name=cmd, description="Ask the bot a question", cog=Agent)
        async def func(ctx: discord.ApplicationContext, question: str):
            async with get_agent(agent_config, config["llms"], config["mcpServers"]) as agent:
                cnt = 0
                first_msg = None
                await ctx.defer()
                async for event in agent.astream(
                    {
                        "messages": [{
                            "role": "user", "content": question
                        }]
                    },
                    stream_mode="values"
                ):
                    if "messages" in event and isinstance(event['messages'][-1], AIMessage) and event['messages'][-1].text():
                        thread = None
                        if cnt >= 1:
                            if cnt == 1:
                                msg = await ctx.send(first_msg.text())
                                await ctx.defer()
                                thread_name = await get_thread_name(agent_config["llm"], config["llms"], question)
                                thread = await msg.create_thread(name=thread_name, auto_archive_duration=60)
                            await thread.send(event['messages'][-1].text())
                            await ctx.defer()
                        else:
                            first_msg = event['messages'][-1]
                        cnt += 1
                if cnt == 1:
                    await ctx.respond(first_msg.text())

            # await ctx.respond(ai_msg.text())
        print(f"Creating slash command for {agent}")
        setattr(Agent, cmd, func)
        bot.add_application_command(getattr(Agent, cmd))
    # Add the cog to the bot
    cog = Agent(bot)
    
    bot.add_cog(cog)