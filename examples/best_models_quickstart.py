#!/usr/bin/env python3
"""
Quick Start Examples for Best Local LLMs (2025 Edition)

This script demonstrates how to use the top-performing open-source models
that can run locally with <64GB RAM.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull models: ollama pull qwen2.5:14b deepseek-coder-v2:16b mixtral:8x7b
3. Install dependencies: pip install ollama requests

Usage:
    python examples/best_models_quickstart.py
"""

import ollama
import json
from typing import Dict, List
import time


class ModelTester:
    """Test different models with various tasks"""
    
    def __init__(self):
        self.models = {
            "qwen2.5:14b": "Qwen2.5 14B - Multilingual Powerhouse",
            "deepseek-coder-v2:16b": "DeepSeek-Coder 16B - Programming Specialist", 
            "mixtral:8x7b": "Mixtral 8x7B - Efficient MoE Model"
        }
        
    def chat_with_model(self, model: str, prompt: str) -> Dict:
        """Send prompt to model and return response with timing"""
        try:
            start_time = time.time()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            
            return {
                "model": model,
                "response": response['message']['content'],
                "response_time": end_time - start_time,
                "tokens_estimated": len(response['message']['content'].split()),
                "success": True
            }
        except Exception as e:
            return {
                "model": model,
                "error": str(e),
                "success": False
            }
    
    def test_multilingual_capabilities(self):
        """Test multilingual understanding and generation"""
        print("🌍 Testing Multilingual Capabilities")
        print("=" * 50)
        
        prompt = """Translate this sentence to Chinese, Spanish, and French:
        "Artificial intelligence is transforming the world."
        
        Then explain in each language why AI is important (2 sentences each)."""
        
        result = self.chat_with_model("qwen2.5:14b", prompt)
        if result["success"]:
            print(f"Model: {self.models['qwen2.5:14b']}")
            print(f"Response Time: {result['response_time']:.2f}s")
            print(f"Response:\n{result['response']}\n")
        else:
            print(f"Error with qwen2.5:14b: {result['error']}\n")
    
    def test_coding_capabilities(self):
        """Test programming and code generation"""
        print("💻 Testing Coding Capabilities")
        print("=" * 50)
        
        prompt = """Write a Python function that implements a binary search algorithm.
        Include:
        1. Proper type hints
        2. Docstring with examples
        3. Error handling
        4. Test cases
        
        Keep it clean and production-ready."""
        
        result = self.chat_with_model("deepseek-coder-v2:16b", prompt)
        if result["success"]:
            print(f"Model: {self.models['deepseek-coder-v2:16b']}")
            print(f"Response Time: {result['response_time']:.2f}s")
            print(f"Response:\n{result['response']}\n")
        else:
            print(f"Error with deepseek-coder-v2:16b: {result['error']}\n")
    
    def test_reasoning_capabilities(self):
        """Test complex reasoning and analysis"""
        print("🧠 Testing Reasoning Capabilities")
        print("=" * 50)
        
        prompt = """A company has 3 departments: Sales (40 people), Engineering (60 people), and Marketing (20 people).
        
        They're planning a team building event with the following constraints:
        - Budget: $15,000
        - Each person costs $125 for the event
        - They want to invite external guests (2 per employee)
        - External guests cost $75 each
        - They need to rent equipment: $500 base + $10 per person
        
        Questions:
        1. What's the total cost if everyone attends?
        2. If they're over budget, what's the maximum number of employees they can invite?
        3. Suggest 3 cost-reduction strategies.
        
        Show your calculations step-by-step."""
        
        result = self.chat_with_model("mixtral:8x7b", prompt)
        if result["success"]:
            print(f"Model: {self.models['mixtral:8x7b']}")
            print(f"Response Time: {result['response_time']:.2f}s")
            print(f"Response:\n{result['response']}\n")
        else:
            print(f"Error with mixtral:8x7b: {result['error']}\n")
    
    def compare_models_same_task(self):
        """Compare how different models handle the same task"""
        print("⚖️  Model Comparison - Same Task")
        print("=" * 50)
        
        prompt = """Design a simple REST API for a book library system.
        Include:
        1. 4-5 main endpoints
        2. HTTP methods and paths
        3. Request/response examples
        4. Basic authentication approach
        
        Keep it concise but complete."""
        
        for model_id in self.models.keys():
            result = self.chat_with_model(model_id, prompt)
            if result["success"]:
                print(f"\n--- {self.models[model_id]} ---")
                print(f"Response Time: {result['response_time']:.2f}s")
                print(f"Estimated Tokens: {result['tokens_estimated']}")
                print(f"Response Preview: {result['response'][:200]}...")
                print("-" * 30)
            else:
                print(f"\n--- {self.models[model_id]} ---")
                print(f"Error: {result['error']}")
                print("-" * 30)
    
    def check_model_availability(self):
        """Check which models are available locally"""
        print("🔍 Checking Model Availability")
        print("=" * 50)
        
        try:
            models_list = ollama.list()
            available_models = {model['name'] for model in models_list['models']}
            
            for model_id, description in self.models.items():
                status = "✅ Available" if model_id in available_models else "❌ Not Found"
                print(f"{status} - {description}")
                
            if not any(model_id in available_models for model_id in self.models.keys()):
                print("\n⚠️  No test models found!")
                print("To install models, run:")
                for model_id in self.models.keys():
                    print(f"  ollama pull {model_id}")
                print()
                return False
                
        except Exception as e:
            print(f"Error checking models: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
            
        return True
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Best Local LLMs Quick Start Test")
        print("=" * 70)
        print()
        
        if not self.check_model_availability():
            return
        
        print()
        self.test_multilingual_capabilities()
        self.test_coding_capabilities()
        self.test_reasoning_capabilities()
        self.compare_models_same_task()
        
        print("✅ All tests completed!")
        print("\nFor more detailed examples, see:")
        print("- notebooks/8.0-best-local-models-examples.ipynb")
        print("- best-local-models-2025.md")


def main():
    """Main execution function"""
    tester = ModelTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()