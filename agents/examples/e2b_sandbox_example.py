"""
Example usage of the E2B Sandbox Agent

This script demonstrates how to use the E2B Sandbox Agent for:
- Creating secure sandbox environments
- Executing code safely
- Managing files in sandboxes
- VM operations

Make sure to set your E2B_API_KEY in your .env file before running.
"""

import sys
import os

# Add parent directory to path to import capibara modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.e2b_sandbox_agent import E2BSandboxAgent


def basic_sandbox_example():
    """Basic example of creating and using a sandbox."""
    print("🚀 Basic E2B Sandbox Example")
    print("-" * 40)

    try:
        # Initialize the agent
        agent = E2BSandboxAgent()
        print("✅ E2B Sandbox Agent initialized")

        # Create a new sandbox
        sandbox_id = agent.create_sandbox("basic_example")
        print(f"✅ Created sandbox: {sandbox_id}")

        # Execute simple Python code
        code = """
print("Hello from E2B Sandbox!")
print(f"Current working directory: {os.getcwd()}")

# Simple calculation
result = 2 + 2
print(f"2 + 2 = {result}")
"""

        result = agent.execute_code(sandbox_id, code)
        if result["success"]:
            print("✅ Code executed successfully!")
            print("📄 Logs:")
            for log in result["logs"]:
                print(f"  {log}")
        else:
            print(f"❌ Execution failed: {result['error']}")

        # List files in the sandbox
        files = agent.list_files(sandbox_id)
        print(f"📁 Files in root directory: {len(files)} items")

        # Close the sandbox
        agent.close_sandbox(sandbox_id)
        print("✅ Sandbox closed")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def file_operations_example():
    """Example of file operations in sandbox."""
    print("\n📁 File Operations Example")
    print("-" * 40)

    try:
        agent = E2BSandboxAgent()
        sandbox_id = agent.create_sandbox("file_ops_example")

        # Create and write to a file
        code = """
# Create a sample data file
data = '''Name,Age,City
Alice,25,New York
Bob,30,San Francisco
Charlie,35,Chicago'''

with open('/tmp/sample_data.csv', 'w') as f:
    f.write(data)

print("Created sample_data.csv")

# Read and process the file
import csv
with open('/tmp/sample_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['Name']} is {row['Age']} years old and lives in {row['City']}")
"""

        result = agent.execute_code(sandbox_id, code)
        if result["success"]:
            print("✅ File operations completed!")
            for log in result["logs"]:
                print(f"  {log}")

        # List files in /tmp
        files = agent.list_files(sandbox_id, "/tmp")
        print(f"📁 Files in /tmp: {[f.get('name', 'unknown') for f in files]}")

        agent.close_sandbox(sandbox_id)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def data_analysis_example():
    """Example of data analysis in sandbox."""
    print("\n📊 Data Analysis Example")
    print("-" * 40)

    try:
        agent = E2BSandboxAgent()
        sandbox_id = agent.create_sandbox("data_analysis_example")

        # Install packages and perform analysis
        code = """
# Generate sample data and perform basic analysis
import json
import statistics

# Sample dataset
sales_data = [
    {"month": "Jan", "sales": 1200, "region": "North"},
    {"month": "Feb", "sales": 1350, "region": "North"},
    {"month": "Mar", "sales": 1100, "region": "North"},
    {"month": "Jan", "sales": 980, "region": "South"},
    {"month": "Feb", "sales": 1050, "region": "South"},
    {"month": "Mar", "sales": 1200, "region": "South"},
]

# Calculate statistics
all_sales = [item["sales"] for item in sales_data]
avg_sales = statistics.mean(all_sales)
max_sales = max(all_sales)
min_sales = min(all_sales)

print(f"Sales Analysis:")
print(f"  Average: ${avg_sales:.2f}")
print(f"  Maximum: ${max_sales}")
print(f"  Minimum: ${min_sales}")

# Group by region
north_sales = [item["sales"] for item in sales_data if item["region"] == "North"]
south_sales = [item["sales"] for item in sales_data if item["region"] == "South"]

print(f"\\nRegional Analysis:")
print(f"  North average: ${statistics.mean(north_sales):.2f}")
print(f"  South average: ${statistics.mean(south_sales):.2f}")

# Save analysis results
results = {
    "total_avg": avg_sales,
    "north_avg": statistics.mean(north_sales),
    "south_avg": statistics.mean(south_sales),
    "data_points": len(sales_data)
}

with open('/tmp/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\\nAnalysis results saved to /tmp/analysis_results.json")
"""

        result = agent.execute_code(sandbox_id, code)
        if result["success"]:
            print("✅ Data analysis completed!")
            for log in result["logs"]:
                print(f"  {log}")

        agent.close_sandbox(sandbox_id)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def multi_sandbox_example():
    """Example of managing multiple sandboxes."""
    print("\n🔄 Multi-Sandbox Example")
    print("-" * 40)

    try:
        agent = E2BSandboxAgent()

        # Create multiple sandboxes
        sandbox1 = agent.create_sandbox("worker_1")
        sandbox2 = agent.create_sandbox("worker_2")

        print(f"✅ Created sandboxes: {agent.list_active_sandboxes()}")

        # Execute different tasks in each sandbox
        task1 = "print('Worker 1: Processing task A'); result_a = sum(range(100)); print(f'Sum 1-99: {result_a}')"
        task2 = "print('Worker 2: Processing task B'); result_b = len([x for x in range(1000) if x % 7 == 0]); print(f'Multiples of 7 under 1000: {result_b}')"

        result1 = agent.execute_code(sandbox1, task1)
        result2 = agent.execute_code(sandbox2, task2)

        print("📊 Results from Worker 1:")
        for log in result1["logs"]:
            print(f"  {log}")

        print("📊 Results from Worker 2:")
        for log in result2["logs"]:
            print(f"  {log}")

        # Close all sandboxes
        closed_count = agent.close_all_sandboxes()
        print(f"✅ Closed {closed_count} sandboxes")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("🏖️ E2B Sandbox Agent Examples")
    print("=" * 50)

    # Run all examples
    basic_sandbox_example()
    file_operations_example()
    data_analysis_example()
    multi_sandbox_example()

    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("- Check the E2B dashboard for usage statistics")
    print("- Explore advanced E2B features like custom templates")
    print("- Integrate sandbox operations into your CapibaraGPT workflows")