# 🗂 Huxley PDF

Chat with your personal PDF docs.

![Huxley PDF](huxleychat_banner.png)

## Overview

**Highlevel overview of this streamlit app by file.**

[Click here to skip to the installation instructions](#installation)

### `Huxley.py`

The `main()` function is responsible for handling the user interface and processing the uploaded PDF file. Here's a breakdown of the code:

1. The `render_header()` function is called to display the header section of the application. It includes the title, description, and an image.

2. The `sidebar()` function is called to display the sidebar section of the application. It includes information about HuxleyPDF, instructions on how to use it, and input fields for the OpenAI API key.

3. The `setup_environment()` function is called to set up the environment. Currently, it only prints a message indicating that the setup is in progress.

4. The `st.file_uploader()` function is used to upload a PDF file. The user is prompted to select a file with the description "Upload your PDF" and the file type filter set to "pdf".

5. The code then fetches a remote PDF file using the `OnlinePDFLoader` class from the Unstructured library. This is commented out for now.

6. If a PDF file is uploaded, the code extracts the text from the PDF using the `PdfReader` class from the PyMuPDF library.

7. The extracted text is split into chunks using the `CharacterTextSplitter` class from the LangChain library. The chunk size is set to 400 characters, and the overlap between chunks is set to 80 characters.

8. The `OpenAIEmbeddings` class is used to create embeddings for the chunks of text.

9. The `FAISS.from_texts()` function is used to create a FAISS index from the chunks of text and their embeddings. This is commented out for now.

10. The user is prompted to enter a question about the PDF using the `st.text_input()` function.

11. If a question is entered, the code retrieves the documents from the FAISS index that are most similar to the user's question using the `similarity_search()` method.

12. The `OpenAI()` class is used to create an instance of the OpenAI API.

13. The `load_qa_chain()` function is used to create a question-answering chain using the OpenAI API and the "stuff" chain type.

14. The `get_openai_callback()` context manager is used to capture the callback information from the OpenAI API.

15. The `chain.run()` method is used to run the question-answering chain on the input documents and the user's question. The response is printed.

16. The response is displayed using the `st.write()` function.

Overall, the code within the `main()` function handles the user interface, processes the uploaded PDF file, and performs a question-answering task using the OpenAI API and the LangChain library.

> References (1)
>
> [1. `huxley.py - 268-313`](https://github.com/fredsiika/huxley-pdf/blob/127dbfd9b80b1e362f137da29ea8bed94ef16e3f/huxley.py#L269)

## Installation

## Troubleshoot
