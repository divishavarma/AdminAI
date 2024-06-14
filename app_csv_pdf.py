import PyPDF2
import csv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
GROQ_API_KEY = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192",
    temperature=0.2
)


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
    ]


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

    # Rest of the code remains the same...

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

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
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

    # Inform the user that processing has ended. You can now chat.
    msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!")
    await msg.send()

    # store the chain in user session
    cl.user_session.set("chain", chain)
    print("Chain initialized and stored in user session")


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    if chain is None:
        print("Chain object is None! Creating a new chain...")
        # Create a new chain if it doesn't exist
        chain = await create_chain()
        # Store the new chain in the user session
        cl.user_session.set("chain", chain)
    else:
        print("Chain object exists")

    # call backs happens asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()
    print('message:', message.content)
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    # return results
    await cl.Message(content=answer, elements=text_elements).send()


async def create_chain():
    # Initialize GROQ chat
    llm_groq = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192",
        temperature=0.2)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        [], embeddings, metadatas=[]
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
