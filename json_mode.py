from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # .env 파일을 로드합니다.


chat = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    response_format={"type": "json_object"}
)

output = chat.invoke(
    [
        SystemMessage(
            content="Extract the 'name' and 'origin' of any companies mentioned in the following statement. Return a JSON list."
        ),
        HumanMessage(
            content="Google was founded in the USA, while Deepmind was founded in the UK"
        ),
    ]
)
print(output.content)
