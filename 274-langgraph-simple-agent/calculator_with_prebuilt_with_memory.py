from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("llama3.2", temperature=0, model_provider="ollama", base_url="http://localhost:11434")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs"
                    )
                ] + state["messages"],
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


from langgraph.prebuilt import ToolNode, tools_condition

tools_node = ToolNode(tools)


from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tools", tools_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    tools_condition
)

agent_builder.add_edge("tools", "llm_call")
conf = {
    "configurable": {"thread_id": 1}
}

agent = agent_builder.compile(checkpointer=checkpointer)


from langchain.messages import HumanMessage


messages = [HumanMessage(content="First operation: Add 3 and 4")]
messages = agent.invoke(input={"messages": messages}, config=conf)
for m in messages["messages"]:
    m.pretty_print()


while True:
    try:
        user_input = input("Give me an arithmetic operation (or 'exit' to quit, \\s to show state): ")
        if user_input.lower() == "exit":
            print("\n\n\nCiao!!....\n\n\n")
            break
        elif user_input.lower() == "\\s":
            print("\n\n\nCurrent state:\n")
            print(agent.get_state(conf))
            print("\n\n\n")
            continue
        messages = [HumanMessage(content=user_input)]
        for event in agent.stream({"messages": messages}, config=conf):
            for v in event.values():
                print(v["messages"][-1].content)
    except KeyboardInterrupt:
        print("\n\n\nCiao!!....\n\n\n")
        break
