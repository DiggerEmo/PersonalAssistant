from dotenv import load_dotenv
import os 
import schedule
import time

from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma

from assistant import PersonalAssistant

def init():
    load_dotenv()
    
    # Initialize the knowledge base
    #knowledge_base = KnowledgeBaseChain(api_key='your_knowledge_base_api_key', db=chroma_db)
    #vector_store = Chroma(
    #    collection_name="context",
    #  #  embedding_function=embeddings,
    #    persist_directory="./chroma_langchain_db"
    #)

    personal_assistant = PersonalAssistant(get_text_generation_pipeline(), get_text_classification_pipeline(), get_vectorstore(), "manuel_ditzig@trimble.com")
    
    # Schedule the method to run every minute
    schedule.every(1).minute.do(personal_assistant.run())

    while True:
        # Run pending tasks
        schedule.run_pending()
        # Sleep for a short time to avoid high CPU usage
        time.sleep(1)

def get_vectorstore():
    # TODO get more meaningful documents, like load them from notion pages or something
    documents = [
        {"id": "1", "text": "Tasks related to the stability of the application 'Transport Assignment' are always TOP priority."},
        {"id": "2", "text": "."}
    ]

    model_id = "all-MiniLM-L6-v2"
    model_path = "./llm/" + model_id
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_path)
    except:
        embeddings = HuggingFaceEmbeddings(model_name=model_id)
        # embeddings(model_path) -> save

    return Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )

def get_model(model_id):
    model_path = "./llm/" + model_id
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.save_pretrained(model_path)

    return tokenizer, model

def get_text_classification_pipeline():
    tokenizer, model = get_model("bert-base-uncased")
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

def get_text_generation_pipeline():
    tokenizer, model = get_model("gpt2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

if __name__ == "__main__":
    init()
