#Importing required libraries
import streamlit as st 
import pandas as pd
import pypdfium2 as pdfium
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from pathlib import Path
import os
import time
import regex as re
import warnings
warnings.filterwarnings('ignore')

import datetime
from io import BytesIO 
from PIL import Image

image = Image.open('KPMG.png')
pi_image = Image.open('logo.png')
st.set_page_config(page_title="Semantic Search", page_icon=pi_image,  layout="wide")

# Setup OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://nprd-pr-genaivang-cns-openai-aihat3.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "4c3a4f93d4bc4daeaafa1190854ec24e"

model_name = "text-davinci-003-aihat3" # "GPT-35-Turbo-aihat3" # for chatGPT
settings = {
    "temperature": 0,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"]
}

prompt = """Answer the following question from source. Answer in detail with example from source. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. Quote the page number & paragraph number in square brackets for each piece of data you use in the answer,e.g. if one of the piece you use comes from paragraphs .02,.03,.04 from page 210 then quote [Paragraphs .02 to .04, Page 210] after that piece of info. If answer has multiple steps put bullet points or numbering and summarise each point with headings.
{question}
{source}
"""

filename = 'C:/Users/dhananjoymondal/Git-Work-Directory/Semantic-Search-Chainlit/data/PCAOB_Audit_Standards.pdf' 
#filename = 'https://github.com/kpmg-us/Audit-GenerativeAI-Hackathon-3-UC2/blob/main/data/PCAOB_Audit_Standards.pdf' 

def get_pdf_data(pdf):
    n_pages = len(pdf)
    data = ''
    for i in range(0, n_pages,9):
        pages = ''
        for j in range(i, min(i+9, n_pages)):
            page = pdf[j]
            textpage = page.get_textpage()
            text_all = textpage.get_text_range()
            pages += text_all

        data += pages

    return data

max_chars = 3000
pdf_ = pdfium.PdfDocument(filename)
pdf_data = get_pdf_data(pdf_)

chunks= []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10, length_function=len)
chunk = text_splitter.create_documents([pdf_data])
chunks.append(chunk)

#Combining all the chunks in list format.
chunks_list = []
for item in chunks:
    for i in item:
        chunks_list.append(i.page_content)

# Put text chunks in a dataframe
chunked_data = pd.DataFrame()
chunked_data['chunks'] = chunks_list
#chunked_data

#Create Embeddings on table chunks
table_chunked_data = pd.DataFrame()
table_chunked_data['embedding'] = chunked_data["chunks"].apply(lambda item : get_embedding(item, engine='text-embedding-ada-002'))

data = pd.DataFrame(columns=['chunks','embedding'])
data['chunks'] = chunked_data['chunks']
data['embedding'] = table_chunked_data['embedding']

def search_docs(df, user_query,  top_n=3, to_print=True):
    '''The query is passed through a function that embeds the query with the ada model and finds the embedding 
       closest to it from the embedded documents.'''
    embedding = get_embedding(user_query, engine='text-embedding-ada-002')
   
    #Hit the right chunks data
    data = df
    #data = pd.DataFrame(columns=['chunks','embedding'])
    
    #Find top n neighbors based on cosine similarity score
    data["similarities"] = data['embedding'].apply(lambda x: cosine_similarity(x, embedding))
    res = (data.sort_values("similarities", ascending=False).head(top_n))
    context, cos_sim = '', 0.0
    for index, row in res.iterrows():
        context = context + row['chunks'] + "\n\n"
        cos_sim += row['similarities']
    context = context.strip().replace("\n",' ').replace('\r','').replace('\xa0',' ')
    return context, (cos_sim)/top_n


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
# Define Streamlit app  
def app():
    st.markdown("""
    <style>
        .st-tabs__container .st-tabs__header {
            font-size: 400px; /* Increase the font size of tab headings */
            font-weight: bold; /* Make tab headings bold */
        }
        .st-tabs__container .st-tabs__content {
            font-size: 380px; /* Increase the font size of tab content */
        }
        .header {
            display: flex;
            justify-content: space-between;
        }
    </style>
    """, unsafe_allow_html=True)

 
    # Set page title  
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Semantic Search")  # Set app heading 
    col2.image(image, width=200)
    
    tab1, tab2 = st.tabs(["Readme", "Ask a question"])
    
    with tab2: 
        col1, col2 = st.columns([1, 1])
        with col1:
            #user_input = st.text_input("Type a new question (e.g. How to respond to potential frauds in financial statements?)")
            user_input = st.text_input("Chat with your AI Audit Assistant")
            print(user_input)
            
            if user_input:
                #once user query is read pass it to search_doc for context generation  
                context = search_docs(data, user_input, top_n=4) 
                formatted_prompt = prompt.format(question=user_input,source=context)  
                
                stream_resp=openai.Completion.create(
                    engine=model_name, prompt=formatted_prompt, stream=True, **settings
                )
                answer = stream_resp.get("choices")[0].get("text").strip()             
            
            st.write(answer)
        
        with col2:  
            st.write("""
                    1. What should I do if I receive information or evidence indicating that the company committed illegal acts?
                    2. The client's inventory count has not been conducted properly. How should we proceed with our audit?
                    3. How to respond to potential frauds in financial statements?
                    4. How should we proceed further with audit if there is no response to confirmations sent?
                    5. What is a qualified opinion? Show an example.
            """)                

            

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], avatar_style = 'bottts', key=str(i))
                message(st.session_state['past'][i], avatar_style = 'big-ears',is_user=True, key=str(i) + '_user')
        
    
