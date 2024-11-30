from dotenv import load_dotenv
import os 
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import BertModel, AutoTokenizer, TextClassificationPipeline
#from langchain_chroma import Chroma
import schedule
import time
from assistant import PersonalAssistant

def init():
    load_dotenv()
    
    # Initialize the language model
    llm = get_huggingface_pipeline()

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

def get_bert_model():
    model_id = "bert-base-uncased"
    model_path = "./llm/bert"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
    
    try:
        model = BertModel.from_pretrained(model_path)
    except:
        model = BertModel.from_pretrained(model_id)
        model.save_pretrained(model_path)

    return tokenizer, model


def get_huggingface_pipeline():
    tokenizer, model = get_bert_model()
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

if __name__ == "__main__":
    init()
