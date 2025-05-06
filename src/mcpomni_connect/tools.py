from typing import Any
import json
import os
from mcpomni_connect.utils import logger

def load_config():
    """Load the servers configuration file"""
    config_path = "servers_config.json"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def get_local_tools():
    """Get available local tools based on configuration"""
    config = load_config()
    local_tools = []
    
    if config.get("localTools", {}).get("enabled", False):
        try:
            from mcpomni_connect.localtools import LOCAL_TOOLS, TOOL_DEFINITIONS
            enabled_tools = config["localTools"].get("tools", [])
            
            for tool_name in enabled_tools:
                if tool_name in LOCAL_TOOLS and tool_name in TOOL_DEFINITIONS:
                    tool_def = TOOL_DEFINITIONS[tool_name]
                    local_tools.append(tool_def)
                else:
                    logger.warning(f"Local tool {tool_name} specified in config but not found in LOCAL_TOOLS")
        except ImportError:
            logger.warning("localtools.py not found or failed to import")
    
    return local_tools

async def list_tools(
    server_names: list[str], sessions: dict[str, Any]
):
    """List all tools including both remote and local tools"""
    tools = []
    
    # Get remote tools
    for server_name in server_names:
        if sessions[server_name]["connected"]:
            try:
                tools_response = await sessions[server_name][
                    "session"
                ].list_tools()
                tools.extend(tools_response.tools)
                # print(f"tools: {tools}")
            except Exception as e:
                logger.info(f"{server_name} Does not support tools")
    # Add local tools
    tools.extend(get_local_tools())
    logger.info(f"total number of tools connected to the client: {len(tools)}")
    return tools


