import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from rag_utils import (
    load_and_process_papers,
    create_vector_db,
    generate_answer,
    gather_user_requirements,
    recommend_similar_papers,
)

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
chat_memory = ConversationBufferMemory(ai_prefix="AI Assistant")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=150)

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Enter the topic or keywords to search for papers:").send()
    response = await cl.AskUserMessage(content="Enter your search query: ", timeout=60).send()
    search_query = response['content'].strip()
    docs = load_and_process_papers(search_query)
    db = create_vector_db(docs, OPENAI_API_KEY)
    cl.user_session.set("search_query", search_query)
    cl.user_session.set("db", db)
    await cl.Message(content="Please provide your specific questions or requirements about the topic.").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    if user_input.lower() == 'exit':
        await cl.Message(content="Conversation ended. Thank you for using the chatbot!").send()
        return

    search_query = cl.user_session.get("search_query")
    db = cl.user_session.get("db")

    user_requirements = await gather_user_requirements(llm, chat_memory, search_query, message)
    answer = generate_answer(user_requirements, db, llm)
    recommendation = recommend_similar_papers(user_requirements, db, llm)
    response = f"Answer: {answer}\n\nRecommended Papers: {recommendation}"
    await cl.Message(content=response).send()