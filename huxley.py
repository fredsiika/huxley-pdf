import os
import time
import base64
import logging
import tempfile
import requests
import tiktoken
import pinecone
from io import BytesIO
import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI # type: ignore
from pdf2image import convert_from_bytes

from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader, OnlinePDFLoader
    
loader = PyPDFLoader('docs/white_paper.pdf')
pages = []

print(loader)

from templates.qa_prompt import QA_PROMPT
from templates.condense_prompt import CONDENSE_PROMPT

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
config = st.set_page_config(page_title='HuxleyPDF | by Fred Siika', page_icon='🗂', layout='wide')

# index = 'huxleypdf'
# openai_api_key=os.environ['OPENAI_API_KEY']

def check_openai_api_key():
    st.info("Please add your OpenAI API key to begin.")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.stop()
        return False
    else:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        st.success("API key set: " + openai_api_key[:5] + "..." + openai_api_key[-5:])
        return True

def check_pinecone_api_key():
    st.info("Please add your Pinecone API key to continue.")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    if not pinecone_api_key:
        st.stop()
        return False
    else:
        os.environ['PINECONE_API_KEY'] = pinecone_api_key
        st.success("API key set: " + pinecone_api_key[:5] + "..." + pinecone_api_key[-5:])
        return True
def check_pinecone_index():
    st.info("Please add your Pinecone index to continue to begin. If you don't have one use the demo `huxleypdf`")
    pinecone_index = st.text_input("Pinecone Index")
    if not pinecone_index:
        st.stop()
        return False
    else:
        os.environ['PINECONE_INDEX'] = pinecone_index
        st.success("Index set: " + pinecone_index)
        return True

def check_pinecone_namespace():
    st.info("Please add your Pinecone namespace to continue. If you don't have one use the demo `ns1`")
    pinecone_namespace = st.text_input("Pinecone Namespace")
    if not pinecone_namespace:
        st.stop()
        return False
    else:
        os.environ['PINECONE_NAMESPACE'] = pinecone_namespace
        st.success("Namespace set: " + pinecone_namespace)
        return True  

def render_header():
   # Start Top Information
    st.title('🗂 HuxleyPDF')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(("### LLM Assisted Custom Knowledgebase "
                        "\n\n"
                        "HuxleyPDF is a Python application that allows you to upload a PDF and ask questions about it using natural language."
                        "\n\n"
                        "#### How it works "
                        "\n\n"
                        "Upload personal docs and Chat with your PDF files with this GPT4-powered app. "
                        "\n\n"
                        "This tool is powered by [OpenAI](https://openai.com)"
                        "[LangChain](<https://langchain.com/>), and [OpenAI](<https://openai.com>) and made by "
                        "[@fredsiika](<https://twitter.com/fredsiika>)."
                        "\n\n"
                        "View Source Code on [Github](<https://github.com/fredsiika/huxley-pdf/blob/main/huxley.py>)"
                    ))
    with col2:
        st.image(image='huxleychat_banner.png', width=300, caption='Tutorial and accompanying documentation coming soon.')
    # End Top Information
    return

# Function to set up the environment
def setup_environment():
    print('Setting up environment')
    # connect_to_pinecone(index)

def connect_to_pinecone(index_name):
    """Connect to Pinecone and return the index."""

    # find API key in console at app.pinecone.io
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') # or 'PINECONE_API_KEY'
    # find ENV (cloud region) next to API key in console
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') # or 'PINECONE_ENVIRONMENT'

    openai_model= 'gpt-3.5-turbo'
    temperature = 0.5
    
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENVIRONMENT  # next to api key in console
    )
    
    model = ChatOpenAI(
        model_name=openai_model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=False
    )  # max temperature is 2 least is 0
    
    # only create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=model.get_sentence_embedding_dimension(),
            metric='cosine'
        )

    # now connect to the index
    print(f"Connecting to Pinecone..\nindex_name: {index_name}")
    index = pinecone.GRPCIndex(index_name)
    
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    
    loader = PyMuPDFLoader("./docs/white_paper.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    query = "Why did the chicken cross the road?"
    docs = docsearch.similarity_search(query)
    print(f'\n{docs[0].page_content}\n')
    
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
            
            Upload personal docs and Chat with your PDF files with this GPT4-powered app. 
            Built with [LangChain](https://docs.langchain.com/docs/), [Pinecone Vector Db](https://pinecone.io/), deployed on [Streamlit](https://streamlit.io)

            ## How to use:
            
            1. Upload a PDF
            2. Ask a question about the PDF
            3. Get an answer about the PDF
            4. Repeat
            
            ## Before you start using HuxleyPDF:
            
            - You need to have an OpenAI API key. You can get one [here](https://api.openai.com/).
            - You need to have a Pinecone API key. You can get one [here](https://www.pinecone.io/).
            - You need to have a Pinecone environment. You can create one [here](https://www.pinecone.io/).
            
            ## How to obtain your OpenAI API key:

            1. Sign in to your OpenAI account. If you do not have an account, [click here](https://platform.openai.com/signup) to sign up.
            
            2. Visit the [OpenAI API keys page.](https://platform.openai.com/account/api-keys)
            open-key-create
        
            ![Step 1 and 2 Create an API Key Screenshot](https://www.usechatgpt.ai/assets/chrome-extension/open-key-create.png)
            
            3. Create a new secret key and copy & paste it into the "API key" input field below.👇🏾
        ''')
        
        st.markdown('''
            ## OpenAI API key
            
            **Tips:**
            
            - The official OpenAI API is more stable than the ChatGPT free plan. However, charges based on usage do apply.
            - Your API Key is saved locally on your browser and not transmitted anywhere else.
            - If you provide an API key enabled with GPT-4, the extension will support GPT-4.
            - Your free OpenAI API key could expire at some point, therefore please check [the expiration status of your API key here.](https://platform.openai.com/account/usage)
            - Access to ChatGPT may be unstable when demand is high for free OpenAI API key.
            
        ''')
        add_vertical_space(5)
        st.write('[HuxleyPDF](https://github.com/fredsiika/huxley-pdf) was made with ❤️ by [Fred](https://github.com/fredsiika)')
        
        st.write(
            "openai_api_key set: ",
            check_openai_api_key()
            # f'<span style="color:green;">{True}</span>' if os.environ.get('OPENAI_API_KEY') else f'<span style="color:red;">{False}</span>'
        )
        st.write(
            "pinecone_api set: ",
            check_pinecone_api_key()
            # True if os.environ.get('PINECONE_API_KEY') == st.secrets['PINECONE_API_KEY'] else False   
        )
        st.write(
            "pinecone_index set set:",
            check_pinecone_index()
            # os.environ.get('PINECONE_INDEX') == st.secrets['PINECONE_INDEX'],
        )
        st.write(
            'pinecone_namespace set: ',
            check_pinecone_namespace()
            # os.environ.get('PINECONE_NAMESPACE') == st.secrets['PINECONE_NAMESPACE'],
        )
        # st.write(
        #     "pinecone_environment set: ",
            
        #     # os.environ.get('PINECONE_ENVIRONMENT') == st.secrets['PINECONE_ENVIRONMENT'],
        # )

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
                # Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace='ns1')
                Pinecone.from_existing_index(index_name='huxleypdf', embedding=embeddings, namespace='ns1')
                st.success("Ingested File!")
            st.session_state["api_key_configured"] = True
    except Exception as e:
        st.error(f"Error while ingesting the files: {str(e)}")
        return None

# Function to display PDF as image on mobile devices
def show_pdf_as_image(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    for image in images:
        st.image(image)
    
# Function to display PDF as iFrame on desktop
def show_pdf_as_iframe(file):
    if file is not None:
        pdf_bytes = file.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        pdf_reader = PdfReader(file)

def main():
    render_header()
    sidebar()
    # setup_environment()
    
    
    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Fetching remote PDFs using Unstructured
    # loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
    # data = loader.load()
    # print(data)
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=400,
            chunk_overlap=80, # I usually set chunk_overlap == 20% of chunk_size
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        
        #TODO:  render image of pdf
        # show_pdf_as_iframe(pdf)

        knowledge_base = Pinecone.from_existing_index(index_name='huxleypdf', embedding=embeddings, namespace='ns1')

        # show user input
        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                
            st.write(response)
            
            #TODO: Add error handling
            
if __name__ == '__main__':
    main()
