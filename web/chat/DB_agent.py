# Tool Agent를 이용한 챗봇 LLM / 파이썬 버전은 3.10 버전

#################################
# 필요한 패키지 다운로드
#################################
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

#################################
# DB 연결
#################################
db = SQLDatabase.from_uri("sqlite:///BingoRoute.db")

#################################
# 로컬 모델 설정
#################################
llm = OllamaLLM(model="gemma3:4b")

#################################
# 툴킷 생성
#################################
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#################################
# 프롬프트 생성 (Action: None 방지)
#################################
prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful travel assistant for the BingoRoute project.\n"
    "You have access to a tourism SQL database and MUST use tools to retrieve facts.\n"
    "Follow this format exactly:\n\n"
    "Thought: Explain your reasoning briefly.\n"
    "Action: choose ONE tool from {tool_names}\n"
    "Action Input: SQL query (plain text)\n"
    "Observation: result from tool\n"
    "Final Answer: your natural language answer to the user.\n\n"
    "⚠️ Rules:\n"
    "- If an SQL query returns an empty result ([] or None), reply '해당 조건에 맞는 장소가 없습니다.'\n"
    "- Never write 'Action: None' or 'Answer:'.\n"
    "- Always end with 'Final Answer:' when you are done.\n"
    "- If you already know the answer, skip directly to 'Final Answer:'.\n\n"
    "Available tools:\n{tools}"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

#################################
# SQL Agent 생성 (프롬프트 적용)
#################################
# 기본 agent 생성
base_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=prompt,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=25,
    max_execution_time=60
)

# AgentExecutor로 한 번 감싸서 로그 차단
agent = AgentExecutor.from_agent_and_tools(
    agent=base_agent.agent,
    tools=toolkit.get_tools(),
    verbose=False,  # verbose 한 번 더 차단
    handle_parsing_errors=True
)

#################################
# 프롬프트 기반 질의 실행
#################################
response = agent.invoke({"input": "장소 3곳 알려줘"})
print("\n 답변:", response["output"])
