from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage , SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests , math
import os
from typing import Any
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt , Command
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

llm = ChatOpenAI()

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")



def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    chunks = splitter.split_documents(chunks , embeddings)
    vector_store = FAISS.from_documents(chunks , embeddings)
    vector_store.save_local(DB_PATH)



def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path = DB_PATH,
        embeddings = embeddings,
        allow_dangerous_deserialization = True
    )
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 4}
    )

    return retriever


@tool
def rag_tool(query : str) -> str:
    """
        This is for the rag purpose , reading  a uploaded file purpose.

    """


    retrieve = get_retriever()
    documents = get_retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the pdf."

    formatted_documents = []

    for index , document in enumerate(documents , start = 1):
        source = document.metadata.get("source" , "Unknown source")
        page = document.metadata.get("page" , "Unnown page")

        formatted_documents.append(
            f"Documet {index}\n"
            f"Source {source}\n"
            f"Page {page}\n"
            f"Content{document.page_content}"
        )

    return "\n\n".join(formatted_documents)

         



search_tool = TavilySearch(
    max_results = 5,
    topic = "general",
    search_depth = "advanced"
)

@tool
def calculator(expression : str) -> str:
    """
    This is for calculating mathematical values.

    """



    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round" : round,
            "min" : min,
            "max" : max,
            "sum" : sum
        }
        result = eval(expression , {"__builtins__": {}} , allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error : {str(e)}"



@tool
def get_stock_price(symbol : str) -> dict:
    """
        This is for checking the stock price.

    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTS&symbol={symbol}&apikey="
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol : str , quantity : int) -> dict:
    """
    This is for purchasin gthe stocks.
    
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol} ? (Yes / No)")

    if isinstance(decision , str) and decision.lower() == "Yes":
        return {
            "status" : "success",
            "message" : f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol" : symbol,
            "quantity" : quantity,
        }

    else:
        return {
            "status" : "cancelled",
            "message" : f"Purchase order placed for {quantity} shares of {symbol} was decined by human.",
            "symbol" : symbol,
            "quantity" : quantity,
        }


@tool
def get_current_weather(location : str) -> str:
    """
    This is for getting the current weather reports.
    
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return(
            "Weather API key is missing."
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q" : location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params = geocoding_params,
            timeout = 10,
        )

        geo_response.raise_for_status()

        locations: list[dict[str , Any]] = geo_response.json()
        
        if not locations:
            return f"Could not find the location : {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name" , location)
        country = locations[0].get("country" , "")
        state = locations[0].get("state" , "")


        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat" : latitude,
            "lon" : longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params = weather_params,
            timeout = 10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind" , {}).get("speed" , "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000 , 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_locations = " , ".join(location_parts)

        return(
            f"Current weather in {display_locations}:\n"
            f"Condition {description.title()}:\n"
            f"Temperature {temperature}:\n"
            f"Feels like {feels_like}:\n"
            f"Humidity {humidity}:\n"
            f"Pressure {pressure}:\n"
            f"Wind Speed {wind_speed}:\n"
            f"Visibility {visibility_km}:\n"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The Openweather API key is invalid or inactive."

        return f"Weather API returned as HTTp error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError , TypeError , ValueError) as error:
        return f"Unexpected weather API response: {error}"
        

tools = [search_tool , get_stock_price , calculator , get_current_weather , rag_tool , purchase_stock]
llm_with_tools = llm.bind_tools(tools)

          
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage] , add_messages]


from langchain_core.messages import SystemMessage

def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use rag_tool for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"

            "- Use search_tool for current events, recent information, or information "
            "that requires an internet search.\n"

            "- Use calculator for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"

            "- Use get_stock_price when the user asks for the current price of a stock.\n"

            "- Use get_current_weather when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required.\n"
            "Do not invent information from the uploaded document.\n"
            "If the user asks about a PDF but no document is available, ask them to upload a PDF.\n"
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages" : [response]}


tool_node = ToolNode(tools)



conn = sqlite3.connect(database = "chatbot.db" , check_same_thread = False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node" , chat_node)
graph.add_node("tools" , tool_node)
graph.add_edge(START , "chat_node")
graph.add_conditional_edges("chat_node" , tools_condition)
graph.add_edge("tools" , "chat_node")
#graph.add_edge("chat_node" , END)

chatbot = graph.compile(checkpointer = checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])
    return list(all_threads)


"""if __name__ == "__main__":
    print("Agentic Chatbot CLI\n")
    print("Type 'exit' to quit.\n")

    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit" , "quit"}:
            print("Goodbye!")
            break

        state = {"messages" : [HumanMessage(content = user_input)]}

        result = chatbot.invoke(
            state,
            config = {"configurable": {"thread_id" : thread_id}},
        )

        interrupts = result.get("__interrupt__" , [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume = decision),
                config = {"configurable" : {"thread_id" : thread_id}},
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")"""


