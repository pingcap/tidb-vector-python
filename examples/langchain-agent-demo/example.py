import os
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from utils import format_docs
from knowledge_base import retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from dotenv import find_dotenv
from loguru import logger

logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler_file = FileCallbackHandler(logfile)
handler_strout = StdOutCallbackHandler()

_ = load_dotenv(find_dotenv())


class ClassEnum(str, Enum):
    album = "album"
    director = "director"
    actor = "actor"
    book = "book"
    songwriter = "songwriter"
    musician = "musician"
    others = "others"


class Classification(BaseModel):
    """Classify the document into a category."""

    # ! Only Hinting category is not work for 'convert method', need to specify the values of the category in desc,
    category: ClassEnum = Field(
        description=f"The category of the document, should be one of the following values: {[e.value for e in ClassEnum]}"
    )
    reason: str = Field(description="The reason for the classification.")


model = ChatOpenAI(
    base_url=os.environ.get('OLLAMA_BASE_URL'),
    api_key=os.environ.get('OLLAMA_API_KEY'),
    # model need support instruction function-calling.
    model=os.environ.get('LM_MODEL_NAME'),
    temperature=0,
)
tools = [convert_to_openai_tool(Classification)]
model_with_tools = model.bind_tools(tools=tools, tool_choice='required')

parser = OpenAIToolsAgentOutputParser()
prompt = PromptTemplate(
    template="""
You are an intelligent assistant, you will receive some documents about input, base on these info,
tasked with classifying items based on their descriptions, use function calling 'Classification'.

related documents: {documents}
input: {input}
""",
    input_variables=["documents", "input"],
)

chain = {"documents": retriever | format_docs, "input": RunnablePassthrough()} | prompt | model_with_tools | parser

resp = chain.invoke(HumanMessage(content="At My Window"), {"callbacks": [handler_file, handler_strout]})
print(resp)

#
if __name__ == '__main__':
    pass
