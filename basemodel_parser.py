from typing import Literal

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # .env 파일을 로드합니다.


class GetCurrentWeather(BaseModel):
    """현재 지역의 날씨를"""

    location: str = Field(
        description="지역의 도시, ex) 대한민국 청주")
    unit: Literal["눈", "폭설", "폭우", "맑음", "흐림", "비"] = Field(
        description="현재 날씨의 상태"
    )


prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)
model = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    tools=[convert_pydantic_to_openai_tool(GetCurrentWeather)]
)
chain = prompt | model | PydanticToolsParser(tools=[GetCurrentWeather])

result = chain.invoke({"input": "현재 괴산의 날씨는 어때?"})
print(result)
