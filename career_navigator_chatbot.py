import streamlit as st
import openai
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv("var.env")

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key"     
def get_completion(prompt, model="gpt=3.5-turbo"):

    messages = [
        SystemMessage(
            content="You're a helpful assistant, Assist in creating a career development plan based on interests and goals"
        ),
        HumanMessage(content=prompt),
    ]

    chat = ChatOpenAI(model="gpt-4-turbo-preview") #lets try gpt4
    return chat.invoke(messages)

st.title("Career Navigator Chatbot")



user_input = st.text_input("How can I help you in your career development?: ")

if st.button("Enter"):
    if user_input:
        response = get_completion(user_input)
        st.text_area("Output:", value=response, height=200)
