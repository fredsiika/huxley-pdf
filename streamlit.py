import os
import time
import pinecone
import tempfile
import tiktoken
import traceback
import configparser
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyMuPDFLoader

from templates.qa_prompt import QA_PROMPT
from templates.condense_prompt import CONDENSE_PROMPT

# Configure environment variables
config = configparser.ConfigParser()
config.read('config.ini')
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Function to set up the environment
def setup_environment():
    index=st.secrets['PINECONE_INDEX']
    st.set_page_config(page_title='HuxleyPDF | by Fred Siika', page_icon='🗂', layout='wide')
    st.title('🗂 HuxleyPDF')
    st.header('Chat With Your PDF Docs')
    connect_to_pinecone(index) 

def connect_to_pinecone(index_name):
    """Connect to Pinecone and return the index."""

    # find API key in console at app.pinecone.io
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') # or 'PINECONE_API_KEY'
    # find ENV (cloud region) next to API key in console
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') # or 'PINECONE_ENVIRONMENT'

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENVIRONMENT  # next to api key in console
    )

    index = pinecone.GRPCIndex(index_name)
    print(f"Connecting to Pinecone..\nindex_name: {index_name}")
    
    # wait a moment for the index to be fully initialized
    time.sleep(1)

    # print(f"\nClients connected to Pinecone index {index_name} \n{index.describe_index_stats()}\n")
    return index.describe_index_stats()


def clear_submit():
    st.session_state["submit"] = False

def sidebar():
    with st.sidebar:
        st.markdown('''## About HuxleyPDF''')
        st.markdown('''    
            HuxleyPDF is a Python application that allows you to upload a PDF and ask questions about it using natural language.
            
            ## How it works:
            
            Upload personal docs and Chat with your PDF files with this GPT4-powered app. Built with LanChain, Pinecone Vector Db, deployed on Streamlit

            ## How to use:
            
            1. Upload a PDF
            2. Ask a question about the PDF
            3. Get an answer about the PDF
            4. Repeat

            ## Before you start using HuxleyPDF:
            - You need to have an OpenAI API key. You can get one [here](https://api.openai.com/).
            - You need to have a Pinecone API key. You can get one [here](https://www.pinecone.io/).
            - You need to have a Pinecone environment. You can create one [here](https://www.pinecone.io/).
        ''')
        st.write(
            "openai_api_key set: ",
            os.environ.get('OPENAI_API_KEY') == st.secrets['OPENAI_API_KEY'],
        )
        st.write(
            "pinecone_api set: ",
            os.environ.get('PINECONE_API_KEY') == st.secrets['PINECONE_API_KEY'],
        )
        st.write(
            "pinecone_environment set: ",
            os.environ.get('PINECONE_ENVIRONMENT') == st.secrets['PINECONE_ENVIRONMENT'],
        )
        st.write(
            "pinecone_index set set:",
            os.environ.get('PINECONE_INDEX') == st.secrets['PINECONE_INDEX'],
        )
        st.write(
            'pinecone_namespace set: ',
            os.environ.get('PINECONE_NAMESPACE') == st.secrets['PINECONE_NAMESPACE'],
        )
        add_vertical_space(5)
        st.write('[HuxleyPDF](https://github.com/fredsiika/huxley-pdf) was made with ❤️ by [Fred](https://github.com/fredsiika)')

def upload_files():
    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type="pdf",
        help="docs, and txt files are still in beta.",
        accept_multiple_files=True,
        on_change=clear_submit
    )
    
    if uploaded_files is None:
        st.info("Please upload a file of type: " + ", ".join(["pdf"]))
    return uploaded_files

# To get the tokenizer corresponding to a specific model in the OpenAI API:
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo') # specific tiktoken encoder which is used by gpt-3.5-turbo: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py#L74

def tiktoken_len(text):
    """Returns the length of the text in tokens."""
    tokens = tokenizer.encode(
        text, 
        disallowed_special=()
    )
    return len(tokens)

# Function to ingest the files
def ingest_files(uploaded_files):
    # find API key in console at app.pinecone.io
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') # or 'PINECONE_API_KEY'
    # find ENV (cloud region) next to API key in console
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') # or 'PINECONE_ENVIRONMENT'

    try:
        with st.spinner("Indexing documents... this might take a while⏳"):
            # Code to ingest the files goes here...
            with tempfile.TemporaryDirectory() as tmpdir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    file_content = uploaded_file.read()
                    st.write("Filename: ", file_name)
                    with open(os.path.join(tmpdir, file_name), "wb") as file:
                        file.write(file_content)
                loader = DirectoryLoader(tmpdir, glob="**/*.pdf", loader_cls=PyMuPDFLoader) # type: ignore
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=tiktoken_len)
                documents = text_splitter.split_documents(documents)
                pinecone.init(
                    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                    environment=PINECONE_ENVIRONMENT  # next to api key in console
                )
                openai_api_key = os.getenv('OPENAI_API_KEY')
                embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key, client=None)
                # Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace='langchain1')
                Pinecone.from_existing_index(index_name='langchain1', embedding=embeddings, namespace='huxley-pdf-embeddings-2023-05-JUNE')
                st.success("Ingested File!")
            st.session_state["api_key_configured"] = True
    except Exception as e:
        st.error(f"Error while ingesting the files: {str(e)}")
        return None

def main():
    try:
        setup_environment()
        sidebar()
        # handle_sidebar()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            openai_api_key = st.text_input("OpenAI API Key", type="password")

        with col2:
            pinecone_api_key = st.text_input("Pinecone API Key", type="password")

        with col3:
            pinecone_environment = st.text_input("Pinecone Environment")

        with col4:
            pinecone_index = st.text_input("Pinecone Index Name")
        
        uploaded_files = upload_files()
       
        # Create a Pinecone index and ingest the documents
        # Pinecone.from_documents(
        #     doc_chunks,
        #     embeddings, 
        #     index_name=pinecone_index, 
        #     namespace='huxley-pdf-embeddings-2023-05-JUNE'
        # )
        index = 'langchain1'
         
        doc = None
        # if uploaded_files:
        #     try:
        #         with st.spinner("Indexing documents... this might take a while⏳"):
                    
        #             # Create a temporary directory to store the uploaded files
        #             with tempfile.TemporaryDirectory() as tmpdir:
        #                 print("Created temporary directory...")
                        
        #                 # Loop through each file that was uploaded
        #                 for uploaded_file in uploaded_files:
        #                     print('Looping through uploaded files...')
                            
        #                     # Get the file name and content
        #                     print("Getting file name and content...")
        #                     file_name = uploaded_file.name
        #                     file_content = uploaded_file.read()
                            
        #                     # Write the file to the temporary directory
        #                     with open(os.path.join(tmpdir, file_name), "wb") as file:
        #                         print("Writing file to temporary directory...\n")
        #                         file.write(file_content)
                        
        #                 # Load the documents from the temporary directory
        #                 loader = DirectoryLoader(tmpdir, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True, silent_errors=True) # type: ignore
        #                 documents = loader.load()
        #                 pages = loader.load_and_split()
        #                 page_no = [page.metadata['page'] for page in pages]
        #                 page_sources = [page.metadata['source'] for page in pages]
        #                 print(f"\n\n{len(pages)}\n\n")
        #                 print(f"\n\n{pages[0]}\n\n")
        #                 print(f"\n\n{pages[:4]}\n\n")
        #                 print('Splitting documents...')
        #                 text_splitter = RecursiveCharacterTextSplitter(
        #                     chunk_size=400,
        #                     chunk_overlap=20, 
        #                     separators=["\n\n", "\n", " ", ""]
        #                 )
        #                     # length_function=len, 
        #                 # documents = text_splitter.split_documents(documents)
        #                 docs_chunks = text_splitter.split_documents(documents)
        #                 pdf_chunks = text_splitter.split_text(docs_chunks[0])
        #                 # print(f"\n\n{tiktoken_len(pdf_chunks)} tokens loaded\n\n")
        #                 # Initializing Pinecone
        #                 print('Initializing Pinecone index...')
        #                 # openai_api_key = os.getenv('OPENAI_API_KEY')
        #                 # pinecone.init(
        #                 #     api_key=pinecone_api_key,  # find at app.pinecone.io
        #                 #     environment=pinecone_environment  # next to api key in console
        #                 # )
        #                 # Create an OpenAI embeddings object
        #                 print('Creating OpenAI embeddings object...')
        #                 uploaded_embeddings = OpenAIEmbeddings(
        #                     model='text-embedding-ada-002', 
        #                     openai_api_key=openai_api_key, 
        #                     client=None
        #                 )
        #                 docsearch = Pinecone.from_existing_index(index_name=index, embedding=uploaded_embeddings, namespace='huxley-pdf-embeddings-2023-05-JUNE')
        #                 query = "What did the president say about Ketanji Brown Jackson"
        #                 docs = docsearch.similarity_search(query)
        #                 st.success(f"\nIngested File Successfully!\n{len(pdf_chunks)} documents ingested!")
        #             st.session_state["api_key_configured"] = True
        #     except Exception as e:
        #         tb = traceback.format_exc()
        #         st.error(f"Error while rendering the response: {str(e)}\n{tb}")
        ingest_files(uploaded_files)
                   
        message = st.text_input('User Input:', on_change=clear_submit)
        temperature = st.slider('Temperature', 0.0, 2.0, 0.7)
        source_amount = st.slider('Sources', 1, 8, 4)
        # message, temperature, source_amount = get_user_input()

        # handle_conversation(message, temperature, source_amount)
        
        button = st.button('Submit')
        # if button or st.session_state.get('submit'):
        #     if not st.session_state.get("api_key_configured"):
        #             st.error("Please configure your OpenAI API key!")
        #     elif not index:
        #         st.error("Please upload a PDF!")
        #     elif not message:
        #         st.error("Please enter a message!")
        #     else:
        #         st.session_state["submit"] = True
        #         # Output columns
        #         answer_col, context_column, source_column = st.columns(3)

        
        if message:
            chat_history = []
            model_name = 'text-embedding-ada-002'
            embeddings = OpenAIEmbeddings(
                model=model_name, 
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                client=None
            )
            
            pinecone.init(api_key=pinecone_api_key,environment=pinecone_environment)
            vectorstore = Pinecone.from_existing_index(index_name=pinecone_index, embedding=embeddings, text_key='text', namespace='huxley-pdf-embeddings-2023-05-JUNE')
            model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY'), streaming=True, client=None) # max temperature is 2 least is 0
            retriever = vectorstore.as_retriever(search_kwargs={"k": source_amount},  qa_template=QA_PROMPT, question_generator_template=CONDENSE_PROMPT) # 9 is the max sources
            qa = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, return_source_documents=True)
            result = qa({"question": message, "chat_history": chat_history})
            answer = result["answer"]
            source_documents = result['source_documents']

            parsed_documents = []
            for doc in source_documents:
                parsed_doc = {
                    "page_content": doc.page_content,
                    "metadata": {
                        "author": doc.metadata.get("author", ""),
                        "creationDate": doc.metadata.get("creationDate", ""),
                        "creator": doc.metadata.get("creator", ""),
                        "file_path": doc.metadata.get("file_path", ""),
                        "format": doc.metadata.get("format", ""),
                        "keywords": doc.metadata.get("keywords", ""),
                        "modDate": doc.metadata.get("modDate", ""),
                        "page_number": doc.metadata.get("page_number", 0),
                        "producer": doc.metadata.get("producer", ""),
                        "source": doc.metadata.get("source", ""),
                        "subject": doc.metadata.get("subject", ""),
                        "title": doc.metadata.get("title", ""),
                        "total_pages": doc.metadata.get("total_pages", 0),
                        "trapped": doc.metadata.get("trapped", "")
                    }
                }
                parsed_documents.append(parsed_doc)
                print(parsed_doc["metadata"]["source"])

            # Display the response in the Streamlit app
            st.write('AI:')
            st.write(answer)
            for doc in parsed_documents:
                st.write(f"Source:", doc["metadata"]["source"])
                st.write(f"Page Number:", doc["metadata"]["page_number"])
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Error while rendering the response: make sure you've entered your correct api/config keys.")

if __name__ == '__main__':
    main()