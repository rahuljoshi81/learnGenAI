#!/Users/rahjosh2/genai/genai/bin/python
import streamlit as st
import xtarfile as tarfile
import tempfile
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

#Open API Keys
OPENAI_API_KEY ="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"
#upload PDF files
st.header("LAAIT: Log Analyzer AI Tool")
#We can upload multiple files as well. Need to write code to handle multiple file.
#
with st.sidebar:
    st.title("Upload Admin-tech file")
    myFile = st.file_uploader("Upload a Admin-tech file and start asking questions related to that log file.", type = "gz")
    

if myFile is not None:
    #Getting uploaded file and writing it to temp dir
    temp_dir = tempfile.mkdtemp()
    filePath = os.path.join(temp_dir, myFile.name)
    with open(filePath, "wb") as f:
        f.write(myFile.getvalue())
    
    # Extracting Admin-tech and appending content
    content = ''
    tar = tarfile.open(filePath, "r:gz")
    for file in tar.getmembers():
        #st.write(file)
        #Checking if file is symLink.If not, appending content from the file. Need to raise exception for Sym link.
        if file.issym():
            st.write("There is symbolic link")
        else:    
            f = tar.extractfile(file)
            if f is not None:
                content = content+ str(f.read())

    #Break texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(content)
    #st.write(chunks)

    #Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Creating Vector store - FAISS
    vector_store = faiss.FAISS.from_texts(chunks,embedding=embeddings)
        
    #get user questions
    user_question = st.text_input("Type your question here")

    #do Similiarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        
        #define LLM
        llm = ChatOpenAI (
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )    
    
        #Generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)


    tar.close()
