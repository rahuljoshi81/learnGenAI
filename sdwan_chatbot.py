#!/Users/rahjosh2/genai/genai/bin/python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


#Open API Keys
OPENAI_API_KEY ="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"
#upload PDF files
st.header("SDWAN Chatbot")

#We can upload multiple files as well. Need to write code to handle multiple file.
#st.file_uploader("Upload multiple files", accept_multiple_files=True)
with st.sidebar:
    st.title("Upload SDWAN related Document")
    myFile = st.file_uploader("Upload a PDF file and start asking questions related to that document.", type = "pdf")
        
#Extract the text

if myFile is not None:
    pdf_reader = PdfReader(myFile)
    #st.write(pdf_reader)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()

#Break texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # embedding openAI
    #initilizing faiss
    #genereting embedding for our chunks and
    #Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    #Creating Vector store - FAISS
    vector_store = faiss.FAISS.from_texts(chunks,embedding=embeddings)
    
    #get user questions
    user_question = st.text_input("Type your question here")

    #do Similiarity search
    # if user question, 
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
        #Output results
        #chain of events
        #Take the question
        #get relevant documents
        #Pass it to the LLM
        #Generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
    


