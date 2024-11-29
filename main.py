from dotenv import load_dotenv
import os 
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
#from langchain_chroma import Chroma
import schedule
import time
from assistant import PersonalAssistant

def init():
    load_dotenv()
    print(os.environ['HF_TOKEN'])
    
    # Initialize the language model
    llm = HuggingFacePipeline(model_name=os.environ['MODEL_NAME'], model_path=os.environ['MODEL_PATH'])

    # Initialize the knowledge base
    #knowledge_base = KnowledgeBaseChain(api_key='your_knowledge_base_api_key', db=chroma_db)
    #vector_store = Chroma(
    #    collection_name="context",
    #  #  embedding_function=embeddings,
    #    persist_directory="./chroma_langchain_db"
    #)

    personal_assistant = PersonalAssistant(llm, "manuel_ditzig@trimble.com")
    
    # Schedule the method to run every minute
    schedule.every(1).minute.do(personal_assistant.run())

    while True:
        # Run pending tasks
        schedule.run_pending()
        # Sleep for a short time to avoid high CPU usage
        time.sleep(1)


if __name__ == "__main__":
    init()
