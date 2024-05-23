# AdminAI

**Overview**
Welcome to the GenAI Chatbot project - AdminAI! This repository hosts a sophisticated AI-powered chatbot leveraging the capabilities of Llama3-8B-8192, capable of engaging in natural language conversations, processing PDF and CSV files, and interfacing with a MySQL database for data retrieval.

**Features**
- Natural Language Interaction: Engage in conversations with the AI in natural language.
- PDF Upload and Chat: Upload PDF files and converse with the AI about their contents.
- CSV Upload and Chat: Upload CSV files and chat with the AI to gain insights from the data.
- Database Connectivity: Connect the chatbot to a MySQL database to query data using natural language commands.
- SQL Query Translation: Convert natural language queries into SQL queries for seamless interaction with the database.
- Natural Language Responses: Receive responses from the database in natural language, enhancing user experience.

**Solution Diagram**
![Diagram1-try](https://github.com/divishavarma/AdminAI/assets/107746948/93e6fa51-def6-48ef-b8f8-2333515c2bd3)


**FlowChart**
![Diagram-try2](https://github.com/divishavarma/AdminAI/assets/107746948/32888498-e46d-4abe-ae2b-c56a8fba9a5f)

**Usage**

**Installation**
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/divishavarma/AdminAI.git
    ```
2. Navigate to the project directory:
    ```bash
    cd AdminAI
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

**Usage**
- Run the application:
    ```bash
    chainlit run app.py
    ```
- Access the chatbot interface through your browser or preferred interface.
- Upload PDF or CSV files to engage in conversations or analyze data.
- Connect the chatbot to your MySQL database for querying data.
