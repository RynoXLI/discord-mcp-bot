import discord
from discord.ext import commands
from discord_mcp_bot.llm import collect_agents, collect_llms

class MCPAgent(commands.Cog): # create a class for our cog that inherits from commands.Cog
    # this class is used to create a cog, which is a module that can be added to the bot

    def __init__(self, bot): # this is a special method that is called when the cog is loaded
        self.bot = bot
        self.config = bot.config
        # self.agents = collect_agents(config)
        self.llms = collect_llms(self.config)

    @discord.slash_command(name="hello", description="Say hello to the bot")
    async def hello(self, ctx: discord.ApplicationContext):
        await ctx.respond("Hey!")
        await ctx.respond("Pong!")

    @discord.slash_command(name="ask", description="Ask the bot a question")
    async def ask(self, ctx: discord.ApplicationContext, question: str):
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", question),
        ]
        ai_msg = self.llms['openai'].invoke(messages)
        await ctx.respond(ai_msg.text())

def setup(bot):
    """
    Setup function to add the cog to the bot.
    
    Args:
        bot (discord.Bot): The Discord bot instance.
    """
    bot.add_cog(MCPAgent(bot))