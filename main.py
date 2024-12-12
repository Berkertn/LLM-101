import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def main():
    """Main function to run the LLM project."""
    # OpenAI API Key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")

    # Initialize OpenAI Chat Model
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)

    # Initialize conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    # Define a dynamic prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Here is the conversation so far:\n{history}\n\nUser: {input}\nAssistant (provide the response as a numbered list):"
    )

    # Create a conversation chain with memory and prompt
    conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt_template)

    print("Welcome to the OpenAI Chat! Type 'exit' to end the chat.")

    user_input = "Could you tell me good company name? it's a IT company"

    chat_history = "\n".join(
        [f"User: {msg.content}" if msg.role == "user" else f"Assistant: {msg.content}"
         for msg in memory.chat_memory.messages]
    )
    prompt = prompt_template.format_prompt(history=chat_history, input=user_input)

    # Generate response based on the prompt
    response = conversation.run(input=prompt.to_string())
    response_lines = [line.strip() for line in response.split("\n") if
                      line.strip().startswith(("1.", "2.", "3.", "4.", "5."))]

    # Listeyi yazdırma
    print("List as a Python list:")
    print(response_lines)
    print(f"AI: {response}")

    # Save user input and AI response in memory
    memory.save_context({"input": user_input}, {"response": response})

if __name__ == "__main__":
    main()
