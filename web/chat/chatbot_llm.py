# Tool Agent를 이용한 챗봇 LLM / 파이썬 버전은 3.10 버전


#################################
# 필요한 패키지 다운로드
#################################
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#################################
# DB 연결
#################################
db = SQLDatabase.from_uri("sqlite:///BingoRoute.db")

#################################
# 로컬 모델 설정
#################################
llm = Ollama(model="gemma3:4b")

#################################
# 툴킷 생성
#################################
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#################################
# 프롬프트 생성
#################################
prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful travel assistant for the BingoRoute project.\n"
    "You have access to a tourism SQL database and MUST use tools to retrieve facts.\n"
    "Follow this ReAct format strictly:\n"
    "Thought: your reasoning\n"
    "Action: one tool from {tool_names}\n"
    "Action Input: the input for that tool (plain SQL only for SQL tools; never wrap in quotes, backticks, or code fences)\n"
    "Observation: result from tool\n\n"
    "Available tools:\n{tools}\n\n"
    "Repeat as needed, and give a clear final answer."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

#################################
# SQL Agent 생성 (프롬프트 적용)
#################################
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=prompt,
    verbose=False,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

#################################
# 프롬프트 기반 질의 실행
#################################
response = agent.invoke({"input": "서울 관광지 중 평점이 제일 높은 1등을 알려줘"})
print(response["output"])
