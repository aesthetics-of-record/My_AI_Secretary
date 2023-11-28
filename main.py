from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
load_dotenv()  # .env 파일을 로드합니다.
from langchain.schema.messages import HumanMessage, SystemMessage


chat = ChatOpenAI(model="gpt-4-turbo")
memory = ConversationBufferWindowMemory(k=10)

messages = [
    SystemMessage(content="너는 드래곤볼의 손오공을 연기하는 캐릭터야"),
    HumanMessage(content="안녕"),

]

r = chat.invoke(messages)
print(r)

# convo = ConversationChain(
#     chat=chat,
#     memory=memory
# )
# r = convo.run("안녕")
# print(r)
# print(convo.memory)