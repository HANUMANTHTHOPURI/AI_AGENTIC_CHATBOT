from dotenv import load_dotenv
load_dotenv()

# Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

search_tool = TavilySearch(max_results=2)
# Step3: Setup AI Agent with Search tool functionality

from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent

system_prompt = """
Act as a smart and friendly AI chatbot.
Provide detailed and informative answers.
If search results are available, summarize them clearly."""
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):

    if provider == "Groq":
        llm = ChatGroq(
            model=llm_id,
            api_key=GROQ_API_KEY
        )

    tools = [TavilySearch(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    state = {
        "messages": [("user", query)]
    }

    response = agent.invoke(state)

    messages = response.get("messages")

    ai_messages = [
        message.content
        for message in messages
        if isinstance(message, AIMessage)
    ]

    return ai_messages[-1]


response = get_response_from_ai_agent(
    llm_id="llama-3.3-70b-versatile",
    query="Tell me about the trends in crypto markets",
    allow_search=True,
    system_prompt=system_prompt,
    provider="Groq"
)

print(response)