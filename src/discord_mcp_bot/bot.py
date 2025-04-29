import discord

class MCPBot(discord.Bot):
    """
    Custom Discord bot class that extends discord.Bot.
    
    This class is used to create a Discord bot with specific configurations and commands.
    """

    def __init__(self, config, description=None, *args, **kwargs):
        """
        Initialize the MCPBot instance.
        
        Args:
            config (dict): Configuration dictionary for the bot.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(description=description, *args, **kwargs)
        self.config = config