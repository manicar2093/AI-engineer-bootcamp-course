# hitl = Human In The Loop
# Thisf file does not contains a usable example of HITL LangGraph implementation, just the 'interrupt_before' argument on
# compile method'

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

def tools_node_wrapper(request, execute):
    print("=== Tools Node Wrapper ===")
    print(request, execute)
    return execute(request)

tools_node = ToolNode(tools, wrap_tool_call=tools_node_wrapper)


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

agent = agent_builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])


from langchain.messages import HumanMessage


messages = [HumanMessage(content="First operation: Add 3 and 4")]
messages = agent.invoke(input={"messages": messages}, config=conf)
print(messages)
for m in messages["messages"]:
    m.pretty_print()

state = agent.get_state(conf)
next_step = state.next
print(f"\n\n\nNext step: {next_step}\n\n\n")
existing_message = state.values["messages"][-1]
print(f"\n\n\nExisting message: {existing_message.tool_calls}\n\n\n")

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
        events = agent.stream({"messages": messages}, config=conf)
        for event in events:
            print("\n\n\nEvent:\n")
            print(event)
            if "__interrupt__" in event:
                print("=== INTERRUPTED BY CONFIGURATION ===")
                raise KeyboardInterrupt

            for v in event.values():
                print(v["messages"][-1].content)
    except KeyboardInterrupt:
        print("\n\n\nCiao!!....\n\n\n")
        break
