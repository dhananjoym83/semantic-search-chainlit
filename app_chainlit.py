import openai
import chainlit as cl

import pandas as pd
import pypdfium2 as pdfium
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
import time
import regex as re
import warnings
warnings.filterwarnings('ignore')

openai.api_type = "azure"
openai.api_base = "https://nprd-pr-genaivang-cns-openai-aihat-3.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "accfd7377fce46ec84084b7815d3183b "

model_name = "text-davinci-003-aihat3"

settings = {
    "temperature": 0.2,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}

prompt = """Answer the following question from source. Answer in detail with example from source. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. Quote the page number & paragraph number in square brackets for each piece of data you use in the answer,e.g. if one of the piece you use comes from paragraphs .02,.03,.04 from page 210 then quote [Paragraphs .02 to .04, Page 210] after that piece of info. If answer has multiple steps put bullet points or numbering and summarise each point with headings.
{question}
{source}
"""
filename = 'https://github.com/kpmg-us/Audit-GenerativeAI-Hackathon-3-UC2/blob/main/data/PCAOB_Audit_Standards.pdf' 

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
chunked_data

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
       
#continuously on a aloop of reading & retrieving user queries i.e. messages 
@cl.on_message
async def main(message: str):
    #once user query is read pass it to search_doc for context generation  
    context = search_docs(data, message, top_n=4) 
    formatted_prompt = prompt.format(question=message,source=context)
    msg = cl.Message(
        content="",
        prompt=formatted_prompt,
        llm_settings=cl.LLMSettings(model_name=model_name, **settings),
    )

    # response = openai.ChatCompletion.create(
    # engine="gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    # messages=[
    #     {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
    #     {"role": "user", "content": "Who were the founders of Microsoft?"}
    # ]
    # )
    async for stream_resp in await openai.Completion.acreate(
        engine=model_name, prompt=formatted_prompt, stream=True, **settings
    ):
        token = stream_resp.get("choices")[0].get("text")
        await msg.stream_token(token)

    await msg.send()
