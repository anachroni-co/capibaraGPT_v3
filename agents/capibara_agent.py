# capibara/agents/capibara_agent.py

import os
import sys

# Get the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

from capibara.jax import jax
from capibara.jax import numpy as jnp
from capibara.jax.experimental import mesh_utils
from capibara.jax.sharding import Mesh, PartitionSpec

# tpu v4-32 imports
from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils,
)
from capibara.jax.tpu_v4.optimizations import (
    create_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
)
from capibara.jax.tpu_v4.profiling import (
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
    def __init__(self, retriever):
        self.retriever = retriever
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)

    def retrieve(self, query):
        with self.memory_monitor:
            return [doc.page_content for doc in self.retriever.invoke(query)]

class CapibaraLLM:
    def __init__(self, model):
        self.model = model
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)
        self.profiler = TpuProfiler()

    def generate(self, prompt):
        with self.memory_monitor:
            with self.profiler:
                return self.model.invoke(prompt)

class CapibaraAgent:
    def __init__(self, name, llm, tools=None, vectordb=None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.vectordb = vectordb
        self.tpu_optimized = True
        self.mesh = create_tpu_mesh((32, 8))
        self.memory_monitor = TpuMemoryMonitor(limit_gb=32)
        self.profiler = TpuProfiler()

    def ask(self, prompt):
        with self.memory_monitor:
            with self.profiler:
                context = ""
                if self.vectordb:
                    context = "\n".join(self.vectordb.retrieve(prompt))
                final_prompt = f"{context}\n\n{prompt}" if context else prompt
                return self.llm.generate(final_prompt)

    def run_tool(self, tool_name, *args, **kwargs):
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
