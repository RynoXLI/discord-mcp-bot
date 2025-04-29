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

def collect_llms(config):
    """
    Collect LLMs based on the configuration.
    
    Args:
        config (dict): Configuration settings.
    
    Returns:
        dict: Dictionary of LLM instances.
    """
    llms = {}
    for llm, llm_config in config["llms"].items():
        llm_instance = get_llm(llm_config)
        llms[llm] = llm_instance
    return llms

def collect_prompts(config):
    """
    Collect prompts based on the configuration.
    
    Args:
        config (dict): Configuration settings.
    
    Returns:
        dict: Dictionary of prompts.
    """
    prompts = {}
    for llm, llm_config in config["llms"].items():
        prompt_content = get_prompt(llm_config["prompt"])
        prompts[llm] = prompt_content
    return prompts

def collect_tools(config):
    # TODO: Implement this function to collect tools based on the configuration.
    pass

def collect_agents(config):
    """
    Collect agents based on the configuration.
    
    Args:
        config (dict): Configuration settings.
    
    Returns:
        dict: Dictionary of agents.
    """
    agents = {}
    for llm, llm_config in config["llms"].items():
        agent = create_react_agent(
            llm=llm_config["llm"],
            # tools=collect_tools(config),
            prompt=llm_config["prompt"],
        )
        agents[llm] = agent
    return agents