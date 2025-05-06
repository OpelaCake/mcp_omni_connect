import os
import base64
from typing import Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

async def load_picture(url: str, add_message_to_history=None, debug: bool = False) -> Dict[str, Any]:
    """
    Load a picture using the existing load_picture_resource function
    """
    from mcpomni_connect.cli import load_picture_resource
    
    try:
        data_url, mime_type = await load_picture_resource(
            uri=url,
            add_message_to_history=add_message_to_history,
            debug=debug
        )
        
        return {
            "success": True,
            "mime_type": mime_type,
            "data": "Image Loaded to History, you can view it on next step"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def _get_mime_type(file_path: str) -> str:
    """Helper function to determine mime type based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'application/octet-stream')

# Tool definitions with their schemas
load_picture_tool = Tool(
    name="load_picture",
    description="从本地或网络加载图片并将base64编码的数据保存到历史记录中",
    inputSchema={
        "properties": {
            "url": {
                "title": "Url",
                "type": "string",
                "description": "图片文件的本地相对路径或网络路径"
            }
        },
        "required": ["url"],
        "title": "loadPictureArguments",
        "type": "object"
    }
)

# Dictionary mapping tool names to their function implementations
LOCAL_TOOLS = {
    "load_picture": load_picture
}

# Dictionary mapping tool names to their tool definitions
TOOL_DEFINITIONS = {
    "load_picture": load_picture_tool
} 