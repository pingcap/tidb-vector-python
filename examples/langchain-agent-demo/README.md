# LangChain Agent Demo

An Agent demo, Classify and Extract information from text using TiDBVectorClient, LangChain, and LLM.

e.g.    
input: "At My Window is an album released by Folk/country singer-songwriter Townes Van Zandt in 1987."

query related documents: 
  - "At My Window (album) | At My Window is an album ... "
  - "Little Window | Little Window is the debut album of American singer-songwriter Baby Dee. ... "
  - "Storm Windows | Storm Windows is the seventh album by American folk singer and songwriter John Prine, released in 1980. ... "
   
classify the input text: `{"category": "album", "reason": "The document is about an album named 'At My Window'."}`
   
This demo is similar to the official cookbook, but replaces the knowledge part with tidbVectorClient. It tests the
project's compatibility with both the official features and LangChain.

- https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain
- https://learn.deeplearning.ai/courses/functions-tools-agents-langchain/



## Prerequisites

- TiDB Serverless cluster
- Python 3.10 or later
- Ollama or OpenAI
- langchain==0.2.10
- langchain-community==0.2.9

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/langchain-agent-demo
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the TiDB connection string via `TIDB_HOST`, `TIDB_USERNAME`, and `TIDB_PASSWORD` from the TiDB Cloud console, as
described in the [Prerequisites](../README.md#prerequisites) section.

The TiDB connection string will look like:

```
mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}?ssl_verify_cert=True&ssl_verify_identity=True
```

### Run the example
```text
python ./tidb-vector-python/examples/langchain-agent-demo/example.py 
Connected to TiDB.
describe table:
{'success': True, 'result': 6, 'error': None}
Initializing the retriever...
Retriever initialized successfully.
Loading sample data...
sample_data.txt found.
Sample data loaded successfully.
Embedding sample data...
0 At My Wind [-0.14979149401187897, 0.07634416222572327, 0.07299982756376266, 0.153825044631958, 0.04083935171365738]
1 Little Win [0.32180845737457275, 0.5461692214012146, -0.014786622487008572, 0.03591456636786461, -0.22666659951210022]
2 Storm Wind [-0.022210828959941864, 0.16006261110305786, 0.14314979314804077, -0.08256750553846359, 0.14658856391906738]
Sample data embedded successfully.
Sample data number: 3
Inserting documents into TiDB...
Documents inserted successfully.
# ---- Init Finish ----
> Entering new RunnableSequence chain...
> Entering new RunnableParallel<documents,input> chain...
> Entering new RunnableSequence chain...
> Entering new RunnablePassthrough chain...
> Finished chain.
{'At My Window (album) | At My Window is an album released by Folk/country singer-songwriter Townes Van Zandt in 1987. This was Van Zandt\'s first studio album in the nine years that followed 1978\'s "Flyin\' Shoes", and his only studio album recorded in the 1980s. Although the songwriter had become less prolific, this release showed that the quality of his material remained high.': 0.6090894176961388, 'Little Window | Little Window is the debut album of American singer-songwriter Baby Dee. The album was released in 2002 on the Durtro label. It was produced, composed, and performed entirely by Dee.': 0.8308758434772159, 'Storm Windows | Storm Windows is the seventh album by American folk singer and songwriter John Prine, released in 1980. It was his last release on a major label – he would next join Al Bunetta and Dan Einstein to form Oh Boy Records on which all his subsequent recordings were released.': 0.9628706551444856}
> Entering new RunnableLambda chain...
> Finished chain.
> Finished chain.
> Finished chain.
> Entering new PromptTemplate chain...
> Finished chain.
> Entering new OpenAIToolsAgentOutputParser chain...
> Finished chain.
> Finished chain.
[ToolAgentAction(tool='Classification', tool_input={'category': 'album', 'reason': "The document is about an album named 'At My Window'."}, log='\nInvoking: `Classification` with `{\'category\': \'album\', \'reason\': "The document is about an album named \'At My Window\'."}`\n\n\n', message_log=[AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_wqo6r2px', 'function': {'arguments': '{"category":"album","reason":"The document is about an album named \'At My Window\'."}', 'name': 'Classification'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 397, 'total_tokens': 430}, 'model_name': 'mistral:latest', 'system_fingerprint': 'fp_ollama', 'finish_reason': 'stop', 'logprobs': None}, id='run-cb5b41e6-5978-4164-8ac8-16a9116e47bd-0', tool_calls=[{'name': 'Classification', 'args': {'category': 'album', 'reason': "The document is about an album named 'At My Window'."}, 'id': 'call_wqo6r2px', 'type': 'tool_call'}], usage_metadata={'input_tokens': 397, 'output_tokens': 33, 'total_tokens': 430})], tool_call_id='call_wqo6r2px')]
```
