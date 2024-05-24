---
title: Simple AI integrations
date: 2024-05-24
---

# Simple AI integrations

The following are simple AI/LLM integrations that can be quickly and easily integrated into day to day work.


```python
from langchain_ollama import ChatOllama
```


```python
llm = ChatOllama(model="llama3.1")
llm.invoke("hey there")
```




    AIMessage(content="How's your day going so far?", additional_kwargs={}, response_metadata={'model': 'llama3.1', 'created_at': '2025-01-22T03:43:17.589958Z', 'done': True, 'done_reason': 'stop', 'total_duration': 803886375, 'load_duration': 32629500, 'prompt_eval_count': 12, 'prompt_eval_duration': 306000000, 'eval_count': 9, 'eval_duration': 463000000, 'message': Message(role='assistant', content='', images=None, tool_calls=None)}, id='run-80e84b62-5fc3-4f3d-9f5c-94962305fade-0', usage_metadata={'input_tokens': 12, 'output_tokens': 9, 'total_tokens': 21})



## Artificial datasets

We will generate an artificial dataset using the LLM to test the additional applications. 

Here a single prompt is used but in cases where a more complex or larger dataset is necessary, a better approach would be to develop this in layers by asking the LLM the to develop various topics, then iterating over the topics to generate as many messages as possible for each topic and finally joining these conversations into a larger dataset.


```python
from IPython.display import display
import json
```


```python
q = """Generate a dataset of 50 messages between coworkers regarding current
        events in an investment office. The messages should discuss 5 different events,
        include 5 different employees, and messages should vary in tone and content. 
        Conversations should refer to specific companies or sectors and not refer to
        broad, vague changes in the economy or the market in general.
        
        The dataset of messages should be returned in a JSON array with each entry 
        including the keys employee_name, content, and datetime.
        
        Please respond with only properly formatted JSON output. Do not add 
        explanation or preamble or any additional text to the response."""
```


```python
response = llm.invoke(q)
```


```python
print(response.content[:200] + "\n...")
```

    ```json
    [
      {
        "employee_name": "John Doe",
        "content": "Just saw that Tesla's stock is up 5% today after their Q2 earnings report. Anyone else following this?",
        "datetime": "2023-02-15T14:3
    ...



```python
messages = json.loads(response.content.replace("```json", "").replace("```", ''))
unique_employees = set([e['employee_name'] for e in messages])

print(f"{len(messages):,} messages generated for {len(unique_employees):,} unique employees")
print()
display(messages[:3])
```

    37 messages generated for 4 unique employees
    



    [{'employee_name': 'John Doe',
      'content': "Just saw that Tesla's stock is up 5% today after their Q2 earnings report. Anyone else following this?",
      'datetime': '2023-02-15T14:30:00'},
     {'employee_name': 'Jane Smith',
      'content': "Yeah, I was surprised by the strength of their sales numbers. Do you think they'll hit their Q4 targets?",
      'datetime': '2023-02-15T14:40:00'},
     {'employee_name': 'John Doe',
      'content': "I'm not so sure... I've been hearing rumors about potential supply chain issues.",
      'datetime': '2023-02-15T14:45:00'}]


## Tagging

Let's tag messages with companies or sectors referenced in the message. This operation can be extremely useful in cases of unknown formats or unknown categories of incoming data.


```python
q = """The following is a message from a company chat. List any companies or industry
        sectors that are referenced in the message. If no companies or sectors are
        referenced, respond with "None".

        Message content: {}
        
        Respond with a comma separated list. Do not include any additional explanation
        or preamble, only respond with a comma separated list of values."""
```


```python
tagged_messages = []
for message in messages:
    response = llm.invoke(q.format(message['content']))
    message['references'] = response.content.split(',')
    tagged_messages.append(message)

    if len(tagged_messages) % 10 == 0:
        print(f"done tagging {len(tagged_messages)} messages")
```

    done tagging 10 messages
    done tagging 20 messages
    done tagging 30 messages



```python
from collections import Counter
import itertools
```


```python
flat_list = list(itertools.chain(*[e['references'] for e in tagged_messages]))
counter = Counter(flat_list)
```


```python
top_tags = counter.most_common()[:5]
top_tags
```




    [('Tesla', 4),
     ('Google', 4),
     (' Technology', 4),
     (' tech industry', 3),
     ('Facebook', 3)]



## Sentiment analysis

We will tag each message with the interpreted sentiment of the author. In such cases of attempting to quantify qualitative data, an LLM quickly analyzing the sentiment in context can be helpful.


```python
import numpy as np
```


```python
q = """The following is a message from a company chat. Evaluate the sentiment expressed
        by the author on a scale from -1 to 1. -1 is the most negative and pessimistic,
        1 is the most positive and optimistic, and 0 is neutral. If you are not able
        to determine a sentiment or mood of the author, return 0.

        Message content: {}
        
        Do not offer any explanation or preamble. Only respod with the number that
        represents the sentiment of the message."""
```


```python
sent_tagged = []
for message in tagged_messages:
    response = llm.invoke(q.format(message['content']))
    message['sentiment'] = float(response.content)
    sent_tagged.append(message)

    if len(sent_tagged) % 10 == 0:
        print(f"done analyzing {len(sent_tagged):,} messages")
```

    done analyzing 10 messages
    done analyzing 20 messages
    done analyzing 30 messages



```python
sentiments = [e['sentiment'] for e in sent_tagged]

mn = np.mean(sentiments)
mdn = np.median(sentiments)
min = np.min(sentiments)
max = np.max(sentiments)

print("Sentiments:")
print(f"Mean: {mn:.2f}")
print(f"Median: {mdn:.2f}")
print(f"Min: {min:.2f}")
print(f"Max: {max:.2f}")
```

    Sentiments:
    Mean: 0.41
    Median: 0.00
    Min: -0.50
    Max: 1.00



```python
for t in [t[0] for t in top_tags]:
    mn = np.mean([e['sentiment'] for e in sent_tagged if t in e['references']])
    print(f"{t.strip()}: {mn:.2f} mean sentiment")
```

    Tesla: 0.38 mean sentiment
    Google: 0.38 mean sentiment
    Technology: 1.00 mean sentiment
    tech industry: 0.17 mean sentiment
    Facebook: 0.00 mean sentiment


## Information extraction

We will extract general information and summaries from the conversation chain. This can offer various insights that might not be previously explored. This methodology can also be applied as a starting point for research and evaluation.


```python
q = """The following is a company chat log. Based upon the conversation content, 
        references and sentiment analysis included, give 3 or more recommendations 
        for future investments. 
        
        Do not examine or comment on the format or structure. Only provide investment
        advice based upon the following content.
        
        Chat log:
        {}
        """
```


```python
response = llm.invoke(q.format(str(sent_tagged)))
```


```python
print(response.content)
```

    This appears to be a dataset of comments from employees in a tech industry company, discussing various news articles and developments related to technology, business, and innovation.
    
    Here are some observations and insights that can be derived from this dataset:
    
    **Most discussed topics:**
    
    1. Technology companies (e.g., Google, Facebook, Amazon)
    2. E-commerce and retail (e.g., Shopify, Amazon)
    3. Electric vehicles and renewable energy (e.g., Tesla, EV industry)
    
    **Sentiment analysis:**
    
    1. The overall sentiment is neutral, with a slight bias towards positive comments (40% positive, 30% negative, 30% neutral).
    2. Employees tend to be cautiously optimistic about the future of technology companies.
    3. There are some concerns and doubts expressed about certain companies' prospects (e.g., Facebook's ad space challenge).
    
    **Employee dynamics:**
    
    1. Three employees (John Doe, Alice Brown, Bob Johnson) account for a significant proportion of comments, suggesting they may be influential or hold leadership positions within the company.
    2. The team has a conversational tone, with frequent mentions of "agreed" and "I'm not sure".
    3. There are some disagreements and debates about specific topics (e.g., Facebook vs. Google), indicating a healthy discussion culture.
    
    **Key themes:**
    
    1. Innovation and disruption in various industries.
    2. Partnership opportunities and collaborations between companies.
    3. Competition and challenges faced by leading technology companies.
    
    These insights provide a glimpse into the company's internal discussions, highlighting its focus on innovation, e-commerce, and renewable energy.


