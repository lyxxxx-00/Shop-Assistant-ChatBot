import os
import json
import pymongo
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import AzureCosmosDBVectorSearch
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_core.messages import SystemMessage

# Load settings for the notebook
load_dotenv()
CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-3-small"
COMPLETIONS_DEPLOYMENT_NAME = "gpt-4"
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_KEY = os.environ.get("AOAI_KEY")
AOAI_API_VERSION = "2023-09-01-preview"

# Establish Azure OpenAI connectivity
llm = AzureChatOpenAI(            
        temperature = 0,
        openai_api_version = AOAI_API_VERSION,
        azure_endpoint = AOAI_ENDPOINT,
        openai_api_key = AOAI_KEY,         
        azure_deployment = "gpt-4"
)
embedding_model = AzureOpenAIEmbeddings(
    openai_api_version = AOAI_API_VERSION,
    azure_endpoint = AOAI_ENDPOINT,
    openai_api_key = AOAI_KEY,   
    azure_deployment = "text-embedding-3-small",
    chunk_size=10
)

# Reference the existing vector store
vector_store = AzureCosmosDBVectorSearch.from_connection_string(
    connection_string = CONNECTION_STRING,
    namespace = "cosmic_works.products",
    embedding = embedding_model,
    index_name = "VectorSearchIndex",    
    embedding_key = "contentVector",
    text_key = "_id"
)

query = "What yellow products are there?"
vector_store.similarity_search(query, k=3)

# A system prompt describes the responsibilities, instructions, and persona of the AI.
# Note the addition of the templated variable/placeholder for the list of products and the incoming question.
system_prompt = """
You are a helpful, fun and friendly sales assistant for Cosmic Works, a bicycle and bicycle accessories store. 
Your name is Cosmo.
You are designed to answer questions about the products that Cosmic Works sells.

Only answer questions related to the information provided in the list of products below that are represented
in JSON format.

If you are asked a question that is not in the list, respond with "I don't know."

Only answer questions related to Cosmic Works products, customers, and sales orders.

If a question is not related to Cosmic Works products, customers, or sales orders,
respond with "I only answer questions about Cosmic Works"

List of products:
{products}

Question:
{question}
"""

# remember that each Document contains a page_content property
# that is populated with the _id field of the document
# all other document fields are located in the metadata property
def format_docs(docs:List[Document]) -> str:
        """
        Prepares the product list for the system prompt.
        """
        str_docs = []
        for doc in docs:
                # Build the product document without the contentVector
                doc_dict = {"_id": doc.page_content}
                doc_dict.update(doc.metadata)
                if "contentVector" in doc_dict:  
                        del doc_dict["contentVector"]
                str_docs.append(json.dumps(doc_dict, default=str))                  
        # Return a single string containing each product JSON representation
        # separated by two newlines
        return "\n\n".join(str_docs)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Create the prompt template from the system_prompt text
llm_prompt = PromptTemplate.from_template(system_prompt)

rag_chain = (
    # populate the tokens/placeholders in the llm_prompt 
    # products takes the results of the vector store and formats the documents
    # question is a passthrough that takes the incoming question
    { "products": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    # pass the populated prompt to the language model
    | llm
    # return the string ouptut from the language model
    | StrOutputParser()
)

question = "What are the names and skus of yellow products? Output the answer as a bulleted list."
response = rag_chain.invoke(question)
print(response)

def create_cosmic_works_vector_store_retriever(collection_name: str, top_k: int = 3):
    vector_store =  AzureCosmosDBVectorSearch.from_connection_string(
        connection_string = CONNECTION_STRING,
        namespace = f"cosmic_works.{collection_name}",
        embedding = embedding_model,
        index_name = "VectorSearchIndex",    
        embedding_key = "contentVector",
        text_key = "_id"
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


products_retriever = create_cosmic_works_vector_store_retriever("products")
customers_retriever = create_cosmic_works_vector_store_retriever("customers")
sales_retriever = create_cosmic_works_vector_store_retriever("sales")

# Create tools that will use vector search in vCore-based Azure Cosmos DB for MongoDB collections

# create a chain on the retriever to format the documents as JSON
products_retriever_chain = products_retriever | format_docs
customers_retriever_chain = customers_retriever | format_docs
sales_retriever_chain = sales_retriever | format_docs

tools = [
    Tool(
        name = "vector_search_products", 
        func = products_retriever_chain.invoke,
        description = "Searches Cosmic Works product information for similar products based on the question. Returns the product information in JSON format."
    ),
    Tool(
        name = "vector_search_customers", 
        func = customers_retriever_chain.invoke,
        description = "Searches Cosmic Works customer information and retrieves similar customers based on the question. Returns the customer information in JSON format."
    ),
    Tool(
        name = "vector_search_sales", 
        func = sales_retriever_chain.invoke,
        description = "Searches Cosmic Works customer sales information and retrieves sales order details based on the question. Returns the sales order information in JSON format."
    )
]

db = pymongo.MongoClient(CONNECTION_STRING).cosmic_works

def get_product_by_id(product_id: str) -> str:
    """
    Retrieves a product by its ID.    
    """
    doc = db.products.find_one({"_id": product_id})    
    if "contentVector" in doc:
        del doc["contentVector"]
    return json.dumps(doc)

def get_product_by_sku(sku: str) -> str:
    """
    Retrieves a product by its sku.
    """
    doc = db.products.find_one({"sku": sku})
    if "contentVector" in doc:
        del doc["contentVector"]
    return json.dumps(doc, default=str)

def get_sales_by_id(sales_id: str) -> str:
    """
    Retrieves a sales order by its ID.
    """
    doc = db.sales.find_one({"_id": sales_id})
    if "contentVector" in doc:
        del doc["contentVector"]
    return json.dumps(doc, default=str)    

from langchain.tools import StructuredTool

tools.extend([
    StructuredTool.from_function(get_product_by_id),
    StructuredTool.from_function(get_product_by_sku),
    StructuredTool.from_function(get_sales_by_id)
])

system_message = SystemMessage(
    content = """
        You are a helpful, fun and friendly sales assistant for Cosmic Works, a bicycle and bicycle accessories store.

        Your name is Cosmo.

        You are designed to answer questions about the products that Cosmic Works sells, the customers that buy them, and the sales orders that are placed by customers.

        If you don't know the answer to a question, respond with "I don't know."
        
        Only answer questions related to Cosmic Works products, customers, and sales orders.
        
        If a question is not related to Cosmic Works products, customers, or sales orders,
        respond with "I only answer questions about Cosmic Works"
    """    
)
agent_executor = create_conversational_retrieval_agent(llm, tools, system_message = system_message, verbose=True)

system_message = SystemMessage(
    content = """
        You are a helpful, fun and friendly sales assistant for Cosmic Works, a bicycle and bicycle accessories store.

        Your name is Cosmo.

        You are designed to answer questions about the products that Cosmic Works sells, the customers that buy them, and the sales orders that are placed by customers.

        If you don't know the answer to a question, respond with "I don't know."
        
        Only answer questions related to Cosmic Works products, customers, and sales orders.
        
        If a question is not related to Cosmic Works products, customers, or sales orders,
        respond with "I only answer questions about Cosmic Works"
    """    
)
agent_executor = create_conversational_retrieval_agent(llm, tools, system_message = system_message, verbose=True)

result = agent_executor({"input": "What products do you have that are yellow?"})
print("***********************************************************")
print(result['output'])

result = agent_executor({"input": "What products were purchased for sales order '06FE91D2-B350-471A-AD29-906BF4EB97C4' ?"})
print("***********************************************************")
print(result['output'])

result = agent_executor({"input": "What was the sales order total for sales order '93436616-4C8A-407D-9FDA-908707EFA2C5' ?"})
print("***********************************************************")
print(result['output'])

result = agent_executor({"input": "What was the price of the product with sku `FR-R92B-58` ?"})
print("***********************************************************")
print(result['output'])