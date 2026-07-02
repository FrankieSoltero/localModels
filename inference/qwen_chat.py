#!/usr/bin/env python3
"""
Interactive chat with Qwen model
Save as: qwen_chat.py
Run with: python qwen_chat.py
"""

from transformers import pipeline
import torch

def load_model():
    """Load the Qwen model"""
    print("Loading Qwen model... This may take a few minutes on first run.")
    
    try:
        pipe = pipeline(
            "text-generation", 
            model="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_response(messages, pipe):
    """Get response from the model"""
    try:
        result = pipe(
            messages, 
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        # Extract just the assistant's response
        full_output = result[0]['generated_text']
        if isinstance(full_output, list):
            # Return the last message (assistant's response)
            return full_output[-1]['content']
        else:
            # If string output, extract the new part after the last user message
            user_content = messages[-1]['content']
            if user_content in full_output:
                response = full_output.split(user_content)[-1].strip()
                return response
            return full_output
            
    except Exception as e:
        return f"Error generating response: {e}"

def print_help():
    """Print available commands"""
    print("\n--- Available Commands ---")
    print("/help    - Show this help message")
    print("/clear   - Clear conversation history")
    print("/history - Show conversation history")
    print("/quit    - Exit the chat")
    print("/exit    - Exit the chat")
    print("-------------------------\n")

def main():
    """Main chat loop"""
    print("🤖 Qwen Interactive Chat")
    print("Type /help for commands, /quit to exit\n")
    
    # Load model
    pipe = load_model()
    if not pipe:
        print("Failed to load model. Exiting.")
        return
    
    # Initialize conversation
    conversation = []
    
    print("Chat started! You can now talk to the model.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == '/help':
                print_help()
                continue
            elif user_input.lower() == '/clear':
                conversation = []
                print("Conversation history cleared.\n")
                continue
            elif user_input.lower() == '/history':
                if not conversation:
                    print("No conversation history.\n")
                else:
                    print("\n--- Conversation History ---")
                    for i, msg in enumerate(conversation, 1):
                        role = msg['role'].title()
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"{i}. {role}: {content}")
                    print("---------------------------\n")
                continue
            elif not user_input:
                print("Please enter a message or command.")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Get model response
            print("Assistant: ", end="", flush=True)
            response = get_response(conversation, pipe)
            print(response)
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})
            
            print()  # Add blank line for readability
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Type /quit to exit properly.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Type /quit to exit or continue chatting.\n")

if __name__ == "__main__":
    main()