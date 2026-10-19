"""
Capibara Agent Module - Core agent implementation with TPU v4-32 support.

This module provides the main agent classes for CapibaraGPT, including
tool wrappers, vector database connectors, LLM interfaces, and the main
agent orchestration class.

Key Components:
    - CapibaraTool: Tool wrapper with TPU optimization and memory monitoring
    - CapibaraVectorDB: Vector database connector for retrieval operations
    - CapibaraLLM: LLM interface with TPU profiling support
    - CapibaraAgent: Main agent class with tool execution and task processing

Example:
    >>> from agents.capibara_agent import CapibaraAgent, CapibaraTool
    >>> tool = CapibaraTool("calculator", lambda x: float(x) * 2)
    >>> agent = CapibaraAgent("assistant", llm, tools=[tool])
    >>> response = agent.ask("What is 2 + 2?")

Author: Skydesk International Dev Team.
"""

import os
import sys

# Get the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

# tpu v4-32 imports
from capibara.jax_ext.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils,
)
from capibara.jax_ext.tpu_v4.optimizations import (
    create_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
)
from capibara.jax_ext.tpu_v4.profiling import (
    TpuProfiler,
    _uniform_fallback_weights,
    _expert_weights_with_cache,
    checkpointed_transformer_block,
)

class CapibaraTool:
    """
    A tool wrapper for Capibara agents with tpu v4-32 optimization support.
    
    Attributes:
        name: The name of the tool
        func: The function to be executed
        tpu_optimized: Whether tpu optimization is enabled
        mesh: tpu mesh configuration
        memory_monitor: Memory monitoring for tpu operations
    """
    
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))  # Configuration for tpu v4-32
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)  # 32GB per chip

    def run(self, *args, **kwargs):
        """
        Execute the tool function with tpu memory monitoring.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function execution
        """
        with self.memory_monitor:
            return self.func(*args, **kwargs)

class CapibaraVectorDB:
    """Vector database connector for retrieval-augmented generation.
    
    This class wraps a retriever and provides TPU-optimized retrieval
    operations with memory monitoring.
    
    Attributes:
        retriever: The underlying retriever object.
        tpu_optimized: Whether TPU optimization is enabled.
        mesh: TPU mesh configuration.
        memory_monitor: Memory monitoring instance.
    """
    
    def __init__(self, retriever):
        """Initialize the vector database connector.
        
        Args:
            retriever: The retriever object for document retrieval.
        """
        self.retriever = retriever
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)

    def retrieve(self, query):
        """Retrieve documents matching the query.
        
        Args:
            query: The search query string.
            
        Returns:
            List of document contents matching the query.
        """
        with self.memory_monitor:
            return [doc.page_content for doc in self.retriever.invoke(query)]

class CapibaraLLM:
    """LLM interface with TPU v4-32 optimization support.
    
    This class wraps an LLM model and provides optimized generation
    with memory monitoring and profiling capabilities.
    
    Attributes:
        model: The underlying LLM model.
        tpu_optimized: Whether TPU optimization is enabled.
        mesh: TPU mesh configuration.
        memory_monitor: Memory monitoring instance.
        profiler: TPU profiler for performance tracking.
    """
    
    def __init__(self, model):
        """Initialize the LLM interface.
        
        Args:
            model: The LLM model to wrap.
        """
        self.model = model
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)
        self.profiler = TpuProfiler()

    def generate(self, prompt):
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt for generation.
            
        Returns:
            The generated response from the model.
        """
        with self.memory_monitor:
            with self.profiler:
                return self.model.invoke(prompt)

class CapibaraAgent:
    """Main agent class for task execution and orchestration.
    
    This class combines LLM, tools, and vector database into a cohesive
    agent capable of answering questions, running tools, and processing
    various task types with TPU optimization.
    
    Attributes:
        name: The agent's name.
        llm: The LLM interface for generation.
        tools: List of available tools.
        vectordb: Optional vector database for retrieval.
        tpu_optimized: Whether TPU optimization is enabled.
        mesh: TPU mesh configuration.
        memory_monitor: Memory monitoring instance.
        profiler: TPU profiler for performance tracking.
    """
    
    def __init__(self, name, llm, tools=None, vectordb=None):
        """Initialize the CapibaraAgent.
        
        Args:
            name: The agent's name.
            llm: The LLM interface instance.
            tools: Optional list of CapibaraTool instances.
            vectordb: Optional CapibaraVectorDB instance.
        """
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.vectordb = vectordb
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)
        self.profiler = TpuProfiler()

    def ask(self, prompt):
        """Ask the agent a question and get a response.
        
        Args:
            prompt: The question or prompt to process.
            
        Returns:
            The generated response from the agent.
        """
        with self.memory_monitor:
            with self.profiler:
                context = ""
                if self.vectordb:
                    context = "\n".join(self.vectordb.retrieve(prompt))
                final_prompt = f"{context}\n\n{prompt}" if context else prompt
                return self.llm.generate(final_prompt)

    def run_tool(self, tool_name, *args, **kwargs):
        """Execute a tool by name.
        
        Args:
            tool_name: The name of the tool to execute.
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            Exception: If the tool is not found.
        """
        with self.memory_monitor:
            with self.profiler:
                for t in self.tools:
                    if t.name == tool_name:
                        return t.run(*args, **kwargs)
                raise Exception(f"Tool '{tool_name}' not found.")
    
    def process(self, input_data):
        """Process input data and return results."""
        try:
            with self.memory_monitor:
                with self.profiler:
                    # Basic processing logic
                    if isinstance(input_data, str):
                        return self.ask(input_data)
                    elif isinstance(input_data, dict):
                        prompt = input_data.get('prompt', '')
                        context = input_data.get('context', '')
                        full_prompt = f"{context}\n{prompt}" if context else prompt
                        return self.ask(full_prompt)
                    else:
                        return {"error": "Unsupported input type"}
        except Exception as e:
            return {"error": str(e)}
    
    def execute(self, task):
        """Execute a task using the agent."""
        try:
            with self.memory_monitor:
                with self.profiler:
                    # Task execution logic
                    if isinstance(task, str):
                        return self.process(task)
                    elif isinstance(task, dict):
                        task_type = task.get('type', 'default')
                        if task_type == 'question':
                            return self.ask(task.get('content', ''))
                        elif task_type == 'tool':
                            tool_name = task.get('tool_name')
                            args = task.get('args', [])
                            kwargs = task.get('kwargs', {})
                            return self.run_tool(tool_name, *args, **kwargs)
                        else:
                            return self.process(task.get('content', ''))
                    else:
                        return {"error": "Invalid task format"}
        except Exception as e:
            return {"error": str(e)}
