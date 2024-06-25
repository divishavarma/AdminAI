import PyPDF2
import csv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import pymysql
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import configparser

from chainlit.input_widget import Slider, Select

# Initialize configparser
config = configparser.ConfigParser()

# Read the configuration file
config.read('config.ini')

db_config = config['database']
host = db_config.get('host')
port = db_config.get('port')
user = db_config.get('user')
password = db_config.get('password')
database = db_config.get('database')

# Load environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
GROQ_API_KEY = os.environ['GROQ_API_KEY']

# Function to initialize the database connection
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    print("DB URI:", db_uri)  # Add this line for debugging
    return SQLDatabase.from_uri(db_uri)

# Function to get the SQL chain
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
  
    llm = ChatGroq(model="Mixtral-8x7b-32768", temperature=0)
  
    def get_schema(_):
        return db.get_table_info()
  
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to get the response based on user query
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
  
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(model="Mixtral-8x7b-32768", temperature=0)
  
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Print the SQL query
    print("SQL Query:", user_query)
    print("SQL Query from LLM:", sql_chain)
  
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="PDF",
            markdown_description="To chat with **PDF Files**.",
            # icon="https://picsum.photos/200",
            accept=["application/pdf"],  # Accept only PDF files
        ),
        cl.ChatProfile(
            name="CSV",
            markdown_description="To chat with **CSV Files**.",
            # icon="https://picsum.photos/250",
            accept=["text/csv"],  # Accept only CSV files
        ),
        cl.ChatProfile(
            name="DataBase",
            markdown_description="To chat with **Database**.",
            # icon="https://picsum.photos/250",
            # accept=["text/csv"],  
        ),
    ]

async def create_chain(model_name="llama3-8b-8192", temperature=0.2, texts=None, metadatas=None):
    # Initialize GROQ chat with selected model and temperature
    llm_groq = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=temperature
    )
    
    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if texts is None:
        texts = []
    if metadatas is None:
        metadatas = []
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

@cl.on_chat_start
async def on_chat_start():
    chat_profile_obj = cl.user_session.get("chat_profile")
    accept_types = []  # Initialize list to store accepted file types

    if chat_profile_obj == "PDF":
        accept_types.append("application/pdf")
    elif chat_profile_obj == "CSV":
        accept_types.append("text/csv")
    else:
        # Handle other chat profiles if needed
        pass

    await cl.Message(
        content=f"starting chat using the {chat_profile_obj} chat profile"
    ).send()

    # Ask the user to choose the model and temperature
    if chat_profile_obj != "DataBase":
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="Select Model",
                    values=["llama3-8b-8192", "Mixtral-8x7b-32768"],
                    initial_index=0,
                    tooltip="Choose the language model you want to use.",
                    description="Please choose the model you want to use for generating responses."
                ),
                Slider(
                    id="Temperature",
                    label="OpenAI - Temperature",
                    initial=0.2,
                    min=0,
                    max=2,
                    step=0.1,
                ),
            ]
        ).send()

        model_name = settings["Model"]
        temperature = settings["Temperature"]

        # Debugging: Print the new model and temperature
        print(f"model: {model_name}, temperature: {temperature}")

    if chat_profile_obj == "DataBase":
        
        db = init_database(user, password, host, port, database)
        cl.user_session.set("db", db)
        
        # Check if 'data' attribute exists, initialize it if not
        if not hasattr(cl.user_session, 'data'):
            cl.user_session.data = {}

        await cl.Message(content="Using model Mixtral-8x7b-32768 with Temperature = 0").send()
        await cl.Message(content="Processing done!").send()
        
        # Initialize chat history if not present
        if "chat_history" not in cl.user_session.data:
            cl.user_session.data["chat_history"] = [
                AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
            ]


    if chat_profile_obj == "CSV" or chat_profile_obj == "PDF":
        files = None  # Initialize variable to store uploaded files

        # Wait for the user to upload files
        while files is None:
            files = await cl.AskFileMessage(
                content=f"Please upload one or more {chat_profile_obj.lower()} files to begin!",
                accept=accept_types,  # Use accepted file types based on the selected chat profile
                max_size_mb=100,  # Optionally limit the file size,
                max_files=10,
                timeout=180,  # Set a timeout for user response,
            ).send()

        # Process each uploaded file
        texts = []
        metadatas = []
        for file in files:
            print(file)  # Print the file object for debugging

            if file.type == "application/pdf":
                # Read the PDF file
                pdf = PyPDF2.PdfReader(file.path)
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()

                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
                file_texts = text_splitter.split_text(pdf_text)
            elif file.type == "text/csv":
                # Read the CSV file
                with open(file.path, 'r') as csvfile:
                    csv_data = csv.DictReader(csvfile)
                    for row in csv_data:
                        file_texts = []
                        for key, value in row.items():
                            file_texts.append(f"{key}: {value}")
                        texts.append("\n".join(file_texts))

            texts.extend(file_texts)

            # Create a metadata for each chunk
            file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
            metadatas.extend(file_metadatas)

        # Store the texts and metadatas in the user session
        cl.user_session.set("texts", texts)
        cl.user_session.set("metadatas", metadatas)
        cl.user_session.set("model_name", model_name)
        cl.user_session.set("temperature", temperature)

        # Inform the user that processing has ended. You can now chat.
        msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!")
        await msg.send()

@cl.on_settings_update
async def setup_agent(settings):
    chat_profile_obj = cl.user_session.get("chat_profile")

    if chat_profile_obj != "DataBase":
        model_name = settings["Model"]
        temperature = settings["Temperature"]

        cl.user_session.set("model_name", model_name)
        cl.user_session.set("temperature", temperature)

        print("Updated Model:", model_name)
        print("Updated Temperature:", temperature)


@cl.on_message
async def main(message: cl.Message):
    chat_profile_obj = cl.user_session.get("chat_profile")
    model_name = cl.user_session.get("model_name", "llama3-8b-8192")
    temperature = cl.user_session.get("temperature", 0.2)

    if chat_profile_obj == "DataBase":
        user_query = message.content
        db = cl.user_session.get("db")
        chat_history = cl.user_session.data.get("chat_history", [])

        chat_history.append(HumanMessage(content=user_query))

        response = get_response(user_query, db, chat_history)
        chat_history.append(AIMessage(content=response))

        await cl.Message(content=response).send()

        cl.user_session.data["chat_history"] = chat_history

    elif chat_profile_obj == "PDF" or chat_profile_obj == "CSV":

        print(f"Updated model: {model_name}, Updated temperature: {temperature}")

        texts = cl.user_session.get("texts", [])
        metadatas = cl.user_session.get("metadatas", [])

        chain = await create_chain(model_name, temperature, texts, metadatas)

        cb = cl.AsyncLangchainCallbackHandler()

        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"

                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

        answer += f"\n\nModel used: {model_name}\nTemperature: {temperature}"

        await cl.Message(content=answer, elements=text_elements).send()

        print('answer:', answer)
