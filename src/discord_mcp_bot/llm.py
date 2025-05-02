import importlib
from langgraph.prebuilt import create_react_agent
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

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

def get_tools(agent_tools, tool_configs) -> list:
    """
    Get the tools based on the configuration.
    
    Args:
        agent_tools (list): List of tools to be used by the agent.
        tool_configs (list): List of tool configurations.
        
    Returns:
        list: Subset of tool configs.
    """
    tools = {}
    for tool in agent_tools:
        tool_cfg = tool_configs.get(tool)
        if tool_cfg:
            tools[tool] = tool_cfg
    return tools

@asynccontextmanager
async def get_agent(agent_config, llm_configs, tool_configs):
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
    mcp_configs = get_tools(agent_config["mcpServers"], tool_configs)
    print(mcp_configs)

    async with MultiServerMCPClient(mcp_configs) as mcp_client:
        agent = create_react_agent(
            model=llm,
            tools=mcp_client.get_tools(),
            prompt=prompt,
        )
        yield agent

async def get_thread_name(llm_name, llm_configs, question):
    """
    Get the thread name based on the LLM name and question.

    Args:
        llm_name (str): Name of the LLM.
        llm_configs (dict): Configuration settings for the LLMs.
        question (str): Question to be asked.

    Returns:
        str: Thread name.
    """
    llm_config = llm_configs.get(llm_name)
    llm = get_llm(llm_config)
    prompt = get_prompt("prompts/thread_name.txt")
    prompt = prompt if prompt.strip() else "You are a helpful assistant that summarizes questions or queries into Discord thread names."

    response = await llm.invoke(
        {
            "messages": [SystemMessage(content=prompt),
                         HumanMessage(content=question)],
        }
    )
    return response.content
