from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.output_parsers import StrOutputParser

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hallo"}]}

graph = StateGraph(state_schema=MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

formater = StrOutputParser()

result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]})
print(result)
print(formater.invoke(result["messages"][-1]))
