"""
Tool Library - Collection of tools for Capibara agents.

This module provides a registry of tools that can be used by Capibara
agents. Tools are wrapped as CapibaraTool instances with TPU optimization
support.

Key Components:
    - tool_map: Dictionary mapping tool names to CapibaraTool instances
    - get_tool_by_name: Function to retrieve a tool by its name

Example:
    >>> from agents.tool_library import get_tool_by_name
    >>> tool = get_tool_by_name("sumar")
    >>> result = tool.run(5, 3)  # Returns 8

Author: Skydesk International Dev Team.
"""

from .capibara_agent import CapibaraTool

def sumar(x, y): 
    """Add two numbers.
    
    Args:
        x: First number.
        y: Second number.
        
    Returns:
        The sum of x and y.
    """
    return x + y

tool_map = {
    """Registry of available tools as CapibaraTool instances.
    
    Add more tools here as needed.
    """
    "sumar": CapibaraTool("sumar", sumar),
    # Agrega more herramientas here...
}

def get_tool_by_name(name):
    """Retrieve a tool by its name from the tool registry.
    
    Args:
        name: The name of the tool to retrieve.
        
    Returns:
        The CapibaraTool instance for the requested tool.
        
    Raises:
        ValueError: If the tool name is not found in the registry.
    """
    if name in tool_map:
        return tool_map[name]
    raise ValueError(f"Tool '{name}' no encontrada")
