---
title: Adaptive RAG
date: 2024-06-01
---

[source](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb)

# Adaptive RAG

Adaptive RAG adapts self-correction with a RAG system. This graph grants the LLM the ability to iterate over retrieval tools until it is satisfied that it has sufficient information to answer the user query.

## 1. Verify LLM


```python
from langchain_ollama import ChatOllama
```


```python
MODEL = "llama3.1"

llm = ChatOllama(model=MODEL)
llm.invoke('hey there').content
```




    "How's it going? Is there something I can help you with or would you just like to chat?"



## 2. Generate index


```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.



```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
```


```python
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
```


```python
# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5",
                              inference_mode="local"),
)
retriever = vectorstore.as_retriever()
```

    Downloading: 100%|██████████| 274M/274M [00:04<00:00, 63.2MiB/s] 
    Verifying: 100%|██████████| 274M/274M [00:00<00:00, 1.08GiB/s]
    Failed to load libllamamodel-mainline-cuda-avxonly.so: dlopen: libcudart.so.11.0: cannot open shared object file: No such file or directory
    Failed to load libllamamodel-mainline-cuda.so: dlopen: libcudart.so.11.0: cannot open shared object file: No such file or directory
    Embedding texts: 100%|██████████| 194/194 [00:39<00:00,  4.97inputs/s]


## 3. Setting up LLMs

### Router


```python
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
```


```python
prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search.
    Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no premable or explanation.
    Question to route: {question}""",
    input_variables=["question"],
)
```


```python
llm = ChatOllama(model=MODEL, format="json", temperature=0)

question_router = prompt | llm | JsonOutputParser()
```

    /tmp/ipykernel_95752/1114416749.py:1: LangChainDeprecationWarning: The class `ChatOllama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import ChatOllama``.
      llm = ChatOllama(model=MODEL, format="json", temperature=0)



```python
question = "llm agent memory"

docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content

print(question_router.invoke({"question": question}))
```

    /tmp/ipykernel_95752/3422603332.py:3: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
      docs = retriever.get_relevant_documents(question)
    Embedding texts: 100%|██████████| 1/1 [00:00<00:00, 66.65inputs/s]


    {'datasource': 'vectorstore'}


### Retrieval grader


```python
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
```


```python
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)
```


```python
llm = ChatOllama(model=MODEL, format="json", temperature=0)

retrieval_grader = prompt | llm | JsonOutputParser()
```


```python
question = "agent memory"

docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content

print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
```

    Embedding texts: 100%|██████████| 1/1 [00:00<00:00, 45.76inputs/s]

    {'score': 'yes'}


    


### Generate


```python
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
```


```python
prompt = hub.pull("rlm/rag-prompt")
```

    /home/jeremy/miniconda3/lib/python3.11/site-packages/langsmith/client.py:241: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(



```python
llm = ChatOllama(model=MODEL, temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()
```


```python
question = "agent memory"

generation = rag_chain.invoke({"context": docs, "question": question})

print(generation)
```

    The correct answer is:
    
    **Maximum Inner Product Search (MIPS)**
    
    This is the external memory component that allows the agent to access and retrieve information from an external vector store, alleviating the restriction of finite attention span.


### Hallucination grader


```python
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)
```


```python
llm = ChatOllama(model=MODEL, format="json", temperature=0)
```


```python
hallucination_grader = prompt | llm | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})
```




    {'score': 'yes'}



### Answer grader


```python
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)
```


```python
llm = ChatOllama(model=MODEL, format="json", temperature=0)
```


```python
answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})
```




    {'score': 'yes'}



### Question rewriter


```python
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)
```


```python
llm = ChatOllama(model=MODEL, temperature=0)
```


```python
question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})
```




    'Here\'s the rewritten question:\n\n**Initial Question:** "What are the details about agent memory?"\n\n**Improved Question:** "Agent memory"\n\nI removed the preamble and made the question more concise, focusing on the specific entity ("agent memory") that we\'re interested in. This should make it easier for a vector store to retrieve relevant information related to "agent memory".'



## 4. Web search tool


```python
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
```


```python
search_ddg = DuckDuckGoSearchRun()
```


```python
search_ddg.invoke("what is langgraph?")
```

## 5. Build graph

### Graph state


```python
from typing import List
from typing_extensions import TypedDict
```


```python
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
```

### Nodes


```python
from langchain.schema import Document
```


```python
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}
```


```python
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
```


```python
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}
```


```python
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
```


```python
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = search_ddg.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}
```

### Edges


```python
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
```


```python
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
```


```python
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
```

### Build graph


```python
from langgraph.graph import END, StateGraph, START
```


```python
graph = StateGraph(GraphState)

# Define the nodes
graph.add_node("web_search", web_search)  # web search
graph.add_node("retrieve", retrieve)  # retrieve
graph.add_node("grade_documents", grade_documents)  # grade documents
graph.add_node("generate", generate)  # generatae
graph.add_node("transform_query", transform_query)  # transform_query

# Build graph
graph.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
graph.add_edge("web_search", "generate")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
graph.add_edge("transform_query", "retrieve")
graph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = graph.compile()
```

# Testing


```python
from pprint import pprint
```


```python
inputs = {"question": "What is the langgraph?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

pprint(value["generation"])
```

    ---ROUTE QUESTION---
    What is the langgraph?
    {'datasource': 'vectorstore'}
    vectorstore
    ---ROUTE QUESTION TO RAG---
    ---RETRIEVE---


    Embedding texts: 100%|██████████| 1/1 [00:00<00:00, 41.41inputs/s]


    "Node 'retrieve':"
    '\n---\n'
    ---CHECK DOCUMENT RELEVANCE TO QUESTION---
    ---GRADE: DOCUMENT NOT RELEVANT---
    ---GRADE: DOCUMENT NOT RELEVANT---
    ---GRADE: DOCUMENT NOT RELEVANT---
    ---GRADE: DOCUMENT NOT RELEVANT---
    ---ASSESS GRADED DOCUMENTS---
    ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---
    "Node 'grade_documents':"
    '\n---\n'
    ---TRANSFORM QUERY---
    "Node 'transform_query':"
    '\n---\n'
    ---RETRIEVE---


    Embedding texts: 100%|██████████| 1/1 [00:00<00:00,  5.52inputs/s]

    "Node 'retrieve':"
    '\n---\n'
    ---CHECK DOCUMENT RELEVANCE TO QUESTION---


    


    ---GRADE: DOCUMENT NOT RELEVANT---
    ---GRADE: DOCUMENT RELEVANT---
    ---GRADE: DOCUMENT RELEVANT---
    ---GRADE: DOCUMENT RELEVANT---
    ---ASSESS GRADED DOCUMENTS---
    ---DECISION: GENERATE---
    "Node 'grade_documents':"
    '\n---\n'
    ---GENERATE---
    ---CHECK HALLUCINATIONS---
    ---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
    ---GRADE GENERATION vs QUESTION---
    ---DECISION: GENERATION ADDRESSES QUESTION---
    "Node 'generate':"
    '\n---\n'
    ('LangGraph definition refers to a library for combining language models with '
     "other components to build applications. It's part of the Prompt Engineering "
     'Guide repo, which contains comprehensive education materials on prompt '
     'engineering. LangChain is another related concept that enables this '
     'functionality.')



```python

```
