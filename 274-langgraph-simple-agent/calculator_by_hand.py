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
    print(f"------------> Invoking LLM with messages: {state['messages']}")
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


from langchain.messages import ToolMessage

def tool_node(state: MessagesState):
    print(f"------------> Invoking tool with calls: {state['messages'][-1].tool_calls}")
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(
            content=observation, tool_call_id=tool_call["id"]
        ))
    return {"messages": result}


from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]

    print(f"------------> Checking if we should continue with messages: {last_message.tool_calls}")

    if last_message.tool_calls:
        return "tools"

    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tools", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue, # Used with should_continue strategy
    ["tools", END] # Used with should_continue strategy
)

agent_builder.add_edge("tools", "llm_call")

agent = agent_builder.compile()


from langchain.messages import HumanMessage


messages = [HumanMessage(content="Add 3 and 4")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()


while True:
    try:
        user_input = input("Give me an arithmetic operation (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("\n\n\nCiao!!....\n\n\n")
            break
        messages = [HumanMessage(content=user_input)]
        for event in agent.stream({"messages": messages}):
            for v in event.values():
                print(v["messages"][-1].content)
    except KeyboardInterrupt:
        print("\n\n\nCiao!!....\n\n\n")
        break
