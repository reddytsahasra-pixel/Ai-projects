from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from google.colab import userdata
import uuid

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=userdata.get("GEMINI_API_KEY")
)

@tool
def search(query: str) -> str:
    return f"GIVE OPTIMAL OUTPUT for: {query}"

agent = create_react_agent(
    model=llm,
    tools=[search],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

user_input = input("Enter your query (or type 'End'): ")

while user_input != "End":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    print(result["messages"][-1].content)
    user_input = input("\nNext query (or End): ")

print("Thanks for chatting!")