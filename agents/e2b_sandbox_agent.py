"""
E2B Sandbox Agent for CapibaraGPT

This module provides capabilities for creating and managing E2B sandboxes
for secure code execution and VM management.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class E2BSandboxAgent:
    """
    Agent for managing E2B sandboxes and code execution.

    This agent provides secure sandbox environments for:
    - Code execution
    - File management
    - VM operations
    - Isolated environment testing
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the E2B Sandbox Agent.

        Args:
            api_key: E2B API key. If not provided, will use E2B_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('E2B_API_KEY')
        if not self.api_key:
            raise ValueError("E2B API key is required. Set E2B_API_KEY environment variable or pass api_key parameter.")

        self.active_sandboxes: Dict[str, Sandbox] = {}

    def create_sandbox(self,
                      sandbox_id: Optional[str] = None,
                      timeout: int = 300,
                      template: Optional[str] = None) -> str:
        """
        Create a new E2B sandbox.

        Args:
            sandbox_id: Optional custom ID for the sandbox
            timeout: Sandbox timeout in seconds (default 5 minutes)
            template: Optional template to use for the sandbox

        Returns:
            str: Sandbox ID
        """
        try:
            # Create sandbox with optional template
            if template:
                sbx = Sandbox.create(template=template, timeout=timeout)
            else:
                sbx = Sandbox.create(timeout=timeout)

            # Use provided ID or generate one
            if not sandbox_id:
                sandbox_id = f"sandbox_{len(self.active_sandboxes)}"

            self.active_sandboxes[sandbox_id] = sbx

            logger.info(f"Created E2B sandbox: {sandbox_id}")
            return sandbox_id

        except Exception as e:
            logger.error(f"Failed to create sandbox: {str(e)}")
            raise

    def execute_code(self,
                    sandbox_id: str,
                    code: str,
                    language: str = "python") -> Dict[str, Any]:
        """
        Execute code in a specific sandbox.

        Args:
            sandbox_id: ID of the sandbox to use
            code: Code to execute
            language: Programming language (default: python)

        Returns:
            Dict containing execution results, logs, and any errors
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        try:
            sbx = self.active_sandboxes[sandbox_id]
            execution = sbx.run_code(code)

            result = {
                "success": True,
                "logs": execution.logs,
                "error": execution.error,
                "results": execution.results,
                "sandbox_id": sandbox_id
            }

            logger.info(f"Code executed in sandbox {sandbox_id}")
            return result

        except Exception as e:
            logger.error(f"Code execution failed in sandbox {sandbox_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "logs": [],
                "results": [],
                "sandbox_id": sandbox_id
            }

    def list_files(self, sandbox_id: str, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files in a sandbox directory.

        Args:
            sandbox_id: ID of the sandbox
            path: Directory path to list (default: root)

        Returns:
            List of file information dictionaries
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        try:
            sbx = self.active_sandboxes[sandbox_id]
            files = sbx.files.list(path)

            logger.info(f"Listed files in {path} for sandbox {sandbox_id}")
            return files

        except Exception as e:
            logger.error(f"Failed to list files in sandbox {sandbox_id}: {str(e)}")
            raise

    def upload_file(self,
                   sandbox_id: str,
                   local_path: str,
                   remote_path: str) -> bool:
        """
        Upload a file to the sandbox.

        Args:
            sandbox_id: ID of the sandbox
            local_path: Local file path
            remote_path: Remote path in sandbox

        Returns:
            bool: Success status
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        try:
            sbx = self.active_sandboxes[sandbox_id]

            with open(local_path, 'rb') as f:
                sbx.files.write(remote_path, f.read())

            logger.info(f"Uploaded {local_path} to {remote_path} in sandbox {sandbox_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload file to sandbox {sandbox_id}: {str(e)}")
            return False

    def download_file(self,
                     sandbox_id: str,
                     remote_path: str,
                     local_path: str) -> bool:
        """
        Download a file from the sandbox.

        Args:
            sandbox_id: ID of the sandbox
            remote_path: Remote path in sandbox
            local_path: Local file path

        Returns:
            bool: Success status
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        try:
            sbx = self.active_sandboxes[sandbox_id]
            content = sbx.files.read(remote_path)

            with open(local_path, 'wb') as f:
                f.write(content)

            logger.info(f"Downloaded {remote_path} from sandbox {sandbox_id} to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download file from sandbox {sandbox_id}: {str(e)}")
            return False

    def close_sandbox(self, sandbox_id: str) -> bool:
        """
        Close and cleanup a sandbox.

        Args:
            sandbox_id: ID of the sandbox to close

        Returns:
            bool: Success status
        """
        if sandbox_id not in self.active_sandboxes:
            logger.warning(f"Sandbox {sandbox_id} not found")
            return False

        try:
            sbx = self.active_sandboxes[sandbox_id]
            sbx.close()
            del self.active_sandboxes[sandbox_id]

            logger.info(f"Closed sandbox {sandbox_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to close sandbox {sandbox_id}: {str(e)}")
            return False

    def close_all_sandboxes(self) -> int:
        """
        Close all active sandboxes.

        Returns:
            int: Number of sandboxes closed
        """
        closed_count = 0
        sandbox_ids = list(self.active_sandboxes.keys())

        for sandbox_id in sandbox_ids:
            if self.close_sandbox(sandbox_id):
                closed_count += 1

        logger.info(f"Closed {closed_count} sandboxes")
        return closed_count

    def get_sandbox_info(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Get information about a sandbox.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            Dict containing sandbox information
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        sbx = self.active_sandboxes[sandbox_id]

        return {
            "sandbox_id": sandbox_id,
            "is_running": True,  # If it's in active_sandboxes, it's running
            "created_at": getattr(sbx, 'created_at', None),
            "template": getattr(sbx, 'template', None)
        }

    def list_active_sandboxes(self) -> List[str]:
        """
        List all active sandbox IDs.

        Returns:
            List of active sandbox IDs
        """
        return list(self.active_sandboxes.keys())

    def __del__(self):
        """Cleanup: close all sandboxes when agent is destroyed."""
        self.close_all_sandboxes()


# Example usage and demonstration
def demo_e2b_sandbox():
    """
    Demonstration of E2B Sandbox Agent capabilities.
    """
    print("E2B Sandbox Agent Demo")
    print("=" * 50)

    try:
        # Initialize agent
        agent = E2BSandboxAgent()

        # Create a sandbox
        sandbox_id = agent.create_sandbox("demo_sandbox")
        print(f"Created sandbox: {sandbox_id}")

        # Execute some Python code
        code = """
import sys
print(f"Python version: {sys.version}")
print("Hello from E2B Sandbox!")

# Create a simple file
with open('/tmp/demo.txt', 'w') as f:
    f.write('This is a demo file created in E2B sandbox')

print("Demo file created successfully")
"""

        result = agent.execute_code(sandbox_id, code)
        print(f"Execution result: {result}")

        # List files in the sandbox
        files = agent.list_files(sandbox_id, "/tmp")
        print(f"Files in /tmp: {files}")

        # Get sandbox info
        info = agent.get_sandbox_info(sandbox_id)
        print(f"Sandbox info: {info}")

        # Close the sandbox
        agent.close_sandbox(sandbox_id)
        print("Sandbox closed successfully")

    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("Make sure to set your E2B_API_KEY environment variable")


if __name__ == "__main__":
    demo_e2b_sandbox()