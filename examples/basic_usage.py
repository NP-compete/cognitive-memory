"""
Basic usage of Cognitive Memory.

This script demonstrates how to remember and recall information
using the core Cognitive Memory features.
"""

import os
import sys

# Assume the package is installed or accessible
try:
    from cognitive_memory import Memory
except ImportError:
    print("Please install cognitive_memory package first: pip install cognitive-memory")
    sys.exit(1)

def main():
    # Initialize a basic memory instance
    # Set COGNITIVE_MEMORY_API_KEY or necessary environment variables if required
    memory = Memory(user_id="demo_user_1")
    
    print("--- Storing Information ---")
    memory.remember("The secret code for the server is 404-ALPHA-9.")
    memory.remember("Alice's favorite color is Cerulean Blue.")
    print("Information stored successfully.\n")
    
    print("--- Recalling Information ---")
    
    # Query for the secret code
    code_result = memory.recall("What is the server secret code?")
    print("Query: What is the server secret code?")
    print(f"Result: {code_result}\n")
    
    # Query for Alice's favorite color
    color_result = memory.recall("What color does Alice like?")
    print("Query: What color does Alice like?")
    print(f"Result: {color_result}\n")

if __name__ == "__main__":
    main()
