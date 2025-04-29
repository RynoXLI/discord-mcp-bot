import importlib
from langgraph.prebuilt import create_react_agent

def get_llm(config):
    """
    Get the LLM instance based on the configuration.
    
    Args:
        config (dict): Configuration settings.
    
    Returns:
        object: LLM instance.
    """
    llm_module = importlib.import_module(config["module"])
    llm_class = getattr(llm_module, config["class"])
    llm_instance = llm_class(**config["params"])
    return llm_instance

def get_prompt(file_path):
    """
    Get the prompt from the specified file.
    Args:
        file_path (str): Path to the prompt file.
    Returns:
        str: Prompt content.
    """
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def get_agent(agent_config, llm_configs, tool_configs):
    """
    Get the agent based on the configuration.

    Args:
        agent_config (dict): Configuration settings for the agent.
        llm_configs (dict): Configuration settings for the LLMs.
        tool_configs (dict): Configuration settings for the tools.

    Returns:
        object: Agent instance.
    """

    llm_config = llm_configs.get(agent_config['llm'])
    llm = get_llm(llm_config)
    prompt = get_prompt(agent_config["prompt"])
    # tools = collect_tools(config["tools"])
    agent = create_react_agent(
        model=llm,
        tools=[],
        prompt=prompt,
        # tools=tools,
        # verbose=True,
    )
    return agent
