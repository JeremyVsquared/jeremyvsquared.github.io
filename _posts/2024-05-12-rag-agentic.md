---
layout: post
title: Agentic RAG
date: 2024-05-12
---

# Agentic RAG

[source](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)

Creeating an agent with a retriever tool for agentic RAG


```python
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
```



```python
def get_model(model='llama3.1', temperature=0):
    if 'gpt' in model:
        return ChatOpenAI(model_name=model, temperature=temperature, streaming=True, 
                          base_url='https://api.runpod.ai/v2/vllm-yj8b3e8ds5vnub/openai/v1')
    else:
        POD_ID = "gv7ylhe0o1dohe"
        return ChatOllama(model='llama3.1:70b', 
                temperature=0, 
                streaming=True, 
                base_url=f'https://{POD_ID}-11434.proxy.runpod.net/')
```

## 1. Retriever

We're starting with a directory of text files which are extracts from PDF files. The PDF files were extracted using PyPDF2 and then Llama 3.1 was used to correct formatting where such errors arose as words being broken across multiple lines, page numbers being inserted into the middle of words and sentences, etc.


```python
import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
```


```python
target_directory = "./"

embedding = OllamaEmbeddings(model="nomic-embed-text")
```


```python
[f for f in os.listdir(target_directory) if '.txt' in f]
```




    ['The Handbook of Fixed Income Securities, 9th Edition.txt']




```python
loader = DirectoryLoader(target_directory, glob="*.txt")
docs = loader.load()
```


```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)
docs_splits = text_splitter.split_documents(docs)
```


```python
chroma = Chroma.from_documents(
    documents=docs_splits,
    embedding=embedding
)
```


```python
retriever = chroma.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "book_rag",
    "Search and return information from a book on fixed income investing",
)

tools = [retriever_tool]
```

## 2.Agent state


```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
```


```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

## 3. Building the graph


```python
from typing import Literal

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.prebuilt import tools_condition
```

### Edges


```python
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = get_model(model='llama3.1')

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
```

### Nodes


```python
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    llm = get_model(model='llama3.1')
    llm_w_tools = llm.bind_tools(tools)
    response = llm_w_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    llm = get_model(model='llama3.1')
    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = get_model(model='llama3.1')

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

```


```python
print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like
```

    ********************Prompt[rlm/rag-prompt]********************


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    ================================[1m Human Message [0m=================================
    
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: [33;1m[1;3m{question}[0m 
    Context: [33;1m[1;3m{context}[0m 
    Answer:


### Graph


```python
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
```


```python
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()
```


```python
from IPython.display import Image, display
try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![png](rag-agentic_files/rag-agentic_26_0.png)
    


## Llama3.1:7b results


```python
import pprint

inputs = {
    "messages": [
        ("user", "how do i evaluate a stock as a good or bad investment opportunity?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
```

    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.1', 'created_at': '2024-12-01T21:49:08.534026824Z', 'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'retrieve_blog_posts', 'arguments': {'query': 'Evaluating stock as good or bad investment opportunity'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 411665987, 'load_duration': 9324354, 'prompt_eval_count': 197, 'prompt_eval_duration': 63427000, 'eval_count': 27, 'eval_duration': 297198000}, id='run-bc115b17-ca95-4d9d-af72-999a7235f66a-0', tool_calls=[{'name': 'retrieve_blog_posts', 'args': {'query': 'Evaluating stock as good or bad investment opportunity'}, 'id': '9d0bc47e-c0b1-4c9e-8ccb-35fb6399f939', 'type': 'tool_call'}], usage_metadata={'input_tokens': 197, 'output_tokens': 27, 'total_tokens': 224})]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='bonds: value, momentum, and low risk. Value Value investing entails buying cheap assets and avoiding expensive ones. This requires a measure of relative valuation. In the case of stock selection, the price of a stock can be compared to its book\n\nvaluation of a particular company, SRI investing involves avoiding a particular investment in a company because it does not meet one’s standards for making positive social impact. An example might be avoiding investments in companies that produce or\n\nand to make investment decisions in the most successful companies.\n\nPortfolio considerations are the framework applied by the portfolio manager to ultimately decide which securities are most appropriate for inclusion in a given portfolio. They create the filter that is used to sceen the potential investment universe', name='retrieve_blog_posts', id='32ecaee1-b2d0-43df-b114-ee2dec104106', tool_call_id='9d0bc47e-c0b1-4c9e-8ccb-35fb6399f939')]}
    '\n---\n'
    ---GENERATE---


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    "Output from node 'generate':"
    '---'
    { 'messages': [ 'To evaluate a stock as a good or bad investment opportunity, '
                    'consider its relative valuation (price vs book value) and '
                    'momentum, while also assessing low-risk characteristics. '
                    "Additionally, think about the company's alignment with your "
                    'values, such as social responsibility (SRI investing). Apply '
                    'portfolio considerations to filter potential investments and '
                    'choose those that align with your goals.']}
    '\n---\n'



```python
import pprint

inputs = {
    "messages": [
        ("user", "how can i consider my portfolio when deciding whether or not to invest in a stock?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
```

    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.1', 'created_at': '2024-12-01T21:50:37.968844042Z', 'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'retrieve_blog_posts', 'arguments': {'query': 'considering portfolio when investing in stocks'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 371358812, 'load_duration': 9665708, 'prompt_eval_count': 200, 'prompt_eval_duration': 55113000, 'eval_count': 24, 'eval_duration': 264019000}, id='run-18d63bcd-a11e-43c9-9081-7d6649be7759-0', tool_calls=[{'name': 'retrieve_blog_posts', 'args': {'query': 'considering portfolio when investing in stocks'}, 'id': '6943c5e3-f9d9-413b-8a4b-0f4607a3ebad', 'type': 'tool_call'}], usage_metadata={'input_tokens': 200, 'output_tokens': 24, 'total_tokens': 224})]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='Portfolio considerations are the framework applied by the portfolio manager to ultimately decide which securities are most appropriate for inclusion in a given portfolio. They create the filter that is used to sceen the potential investment universe\n\ndefault rates may warrant a more defen - sive posture. PORTFOLIO CONSIDERATIONS Portfolio considerations are the framework applied by the investment manag - ers to ultimately decide which securities are most appropriate for inclusion in a given\n\nthe optimal setting to perform the exercise. The investment process of a typical portfolio manager involves different stages. Given the investment universe and objective, the steps usually consist of portfolio construction, risk prediction, and\n\nof investments held in a portfolio but must be evaluated on many critical levels in order to appro - priately manage and lower the overall risk. One of the most common mistakes portfolio managers make is that they “fall in love” with an industry', name='retrieve_blog_posts', id='acb534ef-7a1f-4e40-806e-57816b803a95', tool_call_id='6943c5e3-f9d9-413b-8a4b-0f4607a3ebad')]}
    '\n---\n'
    ---GENERATE---


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    "Output from node 'generate':"
    '---'
    { 'messages': [ 'When considering your portfolio for investing in a stock, '
                    'think about how it aligns with your investment framework and '
                    'objective. Evaluate whether the new stock fits well within '
                    "your existing holdings and doesn't increase overall risk. "
                    'Consider whether you\'re "falling in love" with the industry '
                    'or specific stock, leading to an unbalanced portfolio.']}
    '\n---\n'


## Llama3.1:70b results


```python
import pprint

inputs = {
    "messages": [
        ("user", "how do i evaluate a stock as a good or bad investment opportunity?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
```

    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.1:70b', 'created_at': '2024-12-03T00:28:34.547911437Z', 'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'book_rag', 'arguments': {'query': 'evaluating stocks for investment opportunities'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 7670170749, 'load_duration': 6119050693, 'prompt_eval_count': 181, 'prompt_eval_duration': 306000000, 'eval_count': 23, 'eval_duration': 1243000000}, id='run-a1c38bbe-52c6-4c4f-beda-22d2a4f5b403-0', tool_calls=[{'name': 'book_rag', 'args': {'query': 'evaluating stocks for investment opportunities'}, 'id': '35008cab-b062-439c-a293-59b652057404', 'type': 'tool_call'}], usage_metadata={'input_tokens': 181, 'output_tokens': 23, 'total_tokens': 204})]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='Portfolio considerations are the framework applied by the portfolio manager to ultimately decide which securities are most appropriate for inclusion in a given portfolio. They create the filter that is used to sceen the potential investment universe\n\nthe optimal setting to perform the exercise. The investment process of a typical portfolio manager involves different stages. Given the investment universe and objective, the steps usually consist of portfolio construction, risk prediction, and\n\nstream and at the same time fully reflects the manager’s opportunity set. Consider, for example, a liability funding portfolio that is free to invest in any security in the Bloomberg Barclays Credit Index. An appropriate benchmark for such a\n\nmarkets. Since options prices are primarily determined by the implied volatility of the underlying reference security, a portfolio manager can analyze the opportunity from many different angles.200 0 12/9/2005 2/9/2006 4/9/2006 6/9/2006 8/9/2006', name='book_rag', id='14e3d520-4443-4ce0-be29-9976249d3296', tool_call_id='35008cab-b062-439c-a293-59b652057404')]}
    '\n---\n'
    ---GENERATE---


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    "Output from node 'generate':"
    '---'
    { 'messages': [ 'To evaluate a stock as a good or bad investment opportunity, '
                    "consider the portfolio manager's framework and filter for "
                    'inclusion in a given portfolio. This involves analyzing the '
                    'stock within the context of the investment universe and '
                    'objective, and evaluating its potential impact on portfolio '
                    'construction and risk prediction. Additionally, comparing the '
                    'stock to an appropriate benchmark can also help determine its '
                    'suitability as an investment opportunity.']}
    '\n---\n'



```python
import pprint

inputs = {
    "messages": [
        ("user", "how can i consider my portfolio when deciding whether or not to invest in a stock?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
```

    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.1:70b', 'created_at': '2024-12-03T00:28:44.700526207Z', 'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'book_rag', 'arguments': {'query': 'considering portfolio when deciding whether or not to invest in a stock'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 1884318666, 'load_duration': 39757449, 'prompt_eval_count': 184, 'prompt_eval_duration': 158000000, 'eval_count': 30, 'eval_duration': 1684000000}, id='run-b6e4d542-f824-4cee-a860-a59cbf718105-0', tool_calls=[{'name': 'book_rag', 'args': {'query': 'considering portfolio when deciding whether or not to invest in a stock'}, 'id': '0d04d9e0-c3f1-4919-b285-152956498431', 'type': 'tool_call'}], usage_metadata={'input_tokens': 184, 'output_tokens': 30, 'total_tokens': 214})]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='Portfolio considerations are the framework applied by the portfolio manager to ultimately decide which securities are most appropriate for inclusion in a given portfolio. They create the filter that is used to sceen the potential investment universe\n\ndefault rates may warrant a more defen - sive posture. PORTFOLIO CONSIDERATIONS Portfolio considerations are the framework applied by the investment manag - ers to ultimately decide which securities are most appropriate for inclusion in a given\n\nmarket can position their portfolios to take advantage of the cash-flow reinvestment effect on spreads. T rading Constraints Portfolio managers also should review their main rationales for not trading. Some of the best investment decisions are not\n\nof investments held in a portfolio but must be evaluated on many critical levels in order to appro - priately manage and lower the overall risk. One of the most common mistakes portfolio managers make is that they “fall in love” with an industry', name='book_rag', id='a3b818b9-96e7-49ae-a409-4f6cfb7f79cc', tool_call_id='0d04d9e0-c3f1-4919-b285-152956498431')]}
    '\n---\n'
    ---GENERATE---


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    "Output from node 'generate':"
    '---'
    { 'messages': [ 'When deciding whether or not to invest in a stock, consider '
                    'how it fits into your existing portfolio framework. Think '
                    'about whether the stock aligns with your investment goals and '
                    'risk tolerance, and evaluate its potential impact on the '
                    'overall diversification of your portfolio. Additionally, '
                    'review any trading constraints that may affect your '
                    'decision.']}
    '\n---\n'



```python
import pprint

inputs = {
    "messages": [
        ("user", "what is the best time to invest in a stock?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
```

    ---CALL AGENT---
    "Output from node 'agent':"
    '---'
    { 'messages': [ AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.1:70b', 'created_at': '2024-12-03T00:28:53.72887452Z', 'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'book_rag', 'arguments': {'query': 'best time to invest in a stock'}}}]}, 'done_reason': 'stop', 'done': True, 'total_duration': 1551402510, 'load_duration': 39816290, 'prompt_eval_count': 178, 'prompt_eval_duration': 152000000, 'eval_count': 24, 'eval_duration': 1358000000}, id='run-f352be12-6a57-4923-8240-e68404762a6a-0', tool_calls=[{'name': 'book_rag', 'args': {'query': 'best time to invest in a stock'}, 'id': '8084b9d6-e0a2-40f2-845f-498bf10f5967', 'type': 'tool_call'}], usage_metadata={'input_tokens': 178, 'output_tokens': 24, 'total_tokens': 202})]}
    '\n---\n'
    ---CHECK RELEVANCE---
    ---DECISION: DOCS RELEVANT---
    "Output from node 'retrieve':"
    '---'
    { 'messages': [ ToolMessage(content='versus value stocks, and volatility. A stock may have multiple factors (e.g., both large-cap and growth). To do factor investing properly, you need significant data to run testing of how the factor performed over time. You also need to be able to\n\nare defined and what mix of factors are used is part of what can make each factor strategy unique. Within equities, some of the common investment factors include: size (e.g., large-cap, mid-cap), growth versus value stocks, and volatility. A stock\n\ndrag involved. For nondedicated, opportunistic long-investment investors into the asset class, it implies how much and when to invest. Thus, exposure timing is an important risk retention decision. While the principal sources of portfolio gains for\n\nis\n\nright\n\nit\n\ns\n\na\n\ncrappy\n\ntime\n\nfor\n\nfactor\n\ninvesting. FABOZZI\n\n9E_48.indd   1206FABOZZI\n\n9E_48.indd   1206 4/9/21   3:16 PM4/9/21   3:16 PM\n\nPage 1,236', name='book_rag', id='dc4583cd-dc2b-4c16-bae0-22e7d963a42b', tool_call_id='8084b9d6-e0a2-40f2-845f-498bf10f5967')]}
    '\n---\n'
    ---GENERATE---


    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(


    "Output from node 'generate':"
    '---'
    { 'messages': [ 'The best time to invest in a stock depends on various factors '
                    'such as size (e.g., large-cap, mid-cap), growth versus value '
                    'stocks, and volatility. There is no specific answer provided '
                    'in the context about the optimal timing for investing in a '
                    'stock. Exposure timing is an important risk retention '
                    'decision, but the ideal time is not specified.']}
    '\n---\n'
