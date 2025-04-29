# Discord MCP Bot

## Project Objective

MCP Servers are popping up quickly all throughout industry. With this in mind, this discord bot aims to let users control an llm that has access to MCP servers. The goal is to keep this code very simple, most additionally functionality can be achived by adding in new MCP servers. All you need to do is to edit the `configuration.yml` file to add things like the discord bot, the llm in use, and the MCP servers.

## TODO
 - Setup dockerfile / docker compose
    - Setup common file (OpenAI, Anthropic, Bedrock, Azure, etc)
    - Setup Ollama/Hugginface file
 - Setup documentation / documentation pipeline
    - Add configuration examples
 - Setup ReAct agents
 - Add ability to talk on threads
 - Add MCP tool calls
    - Add ability to approve tool calls

