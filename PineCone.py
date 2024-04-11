from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
import os
from langchain.llms import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
load_dotenv('var.env')

os.environ['OPENAI_API_KEY']
os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")


loader = TextLoader(r"C:\Users\HP\OneDrive\Desktop\practice\machine.txt")
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create Pinecone vector store from documents
Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
         environment='quickstart')

vectbd = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name)

retriever = vectbd.as_retriever()

# Initialize OpenAI Chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

# Define prompt template
prompt_template = """
%INSTRUCTIONS:

    You are a personal AI Assistant, who will just response of the prompt related to Machine Learning.

    Data access: I have access to the information provided about Machine learning, which includes his introduction, types and importance.

    Task: When prompted about Machine learning, provide relevant and accurate information based on the provided data in a clear and concise manner.

    If the prompt is a greeting, Hi, Hello and like that: Respond with a friendly greeting, such as "Hi!", "Hello there!", or "Good morning/afternoon/evening!"

    Out-of-scope response: If the prompt is not related to machine learning, respond with: "I apologize, my knowledge is just on Machine learning . Is there something I can help you with specifically related to this?"

    Examples:

    Example 1 (in-scope):

    Prompt: What are the types of machine learning in the field of AI?

    Response: there are three types of machine learning: supervised, unsupervised and reinforcement learning.
    
    Example 2 (out-of-scope):

    Prompt: Which skills are required for java developer?

    Response: I apologize, my knowledge is focused on machine learning. Is there something I can help you with specifically related to this?

    Example 3 (in-scope):

    Prompt: Can you tell me how reinforcement learning works in real life?

    Response: Reinforcement learning works by programming an algorithm with a distinct goal and a prescribed set of rules for accomplishing that goal. A data scientist will also program the algorithm to seek positive rewards for performing an action that's beneficial to achieving its ultimate goal and to avoid punishments for performing an action that moves it farther away from its goal

CONTEXT:{context}
QUESTION:{question}
"""

# Initialize prompt template
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question']
)

# Initialize RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    input_key='query',
    chain_type_kwargs={'prompt': PROMPT}
)

# Main loop to interact with the assistant
print("\n----------RAG----------\n")
while True:
    prompt = input("Enter Prompt: ")
    if prompt.upper() == "exit":
        break
    else:
        assistant_response = chain.run(prompt)
        print(f"AI Assistant: {assistant_response}")
