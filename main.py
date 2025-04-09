from dotenv import load_dotenv
import os 
import time
import numpy as np
import evaluate
from categories import Category 

from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datasets import load_dataset

from assistant import PersonalAssistant

def init():
    load_dotenv()
 
    print("Task Management Agent")

    personal_assistant = PersonalAssistant(get_text_generation_pipeline(), get_text_classification_pipeline(), get_vectorstore(), "manuel_ditzig@trimble.com")
    
    # Schedule the method to run every minute
    #schedule.every(1).minute.do(personal_assistant.run())

    while True:
        # Sleep for a short time to avoid high CPU usage
        time.sleep(1)
        
        command = input(">> ").strip().lower()

        if command == "exit":
            print("Exiting Task Management Agent. Goodbye!")
            break
        elif command.startswith("add task "):
            task = command[9:]
            personal_assistant.add_task(task)
        elif command == "next task":
            personal_assistant.get_next_task()
        elif command.startswith("complete "):
            task = command[9:]
            personal_assistant.complete_task(task)
        else:
            personal_assistant.take_input(command)

def get_vectorstore():
    # TODO get more meaningful documents, like load them from notion pages or something
    init_document = Document(
        page_content="Tasks related to the stability of the application 'Transport Assignment' are always TOP priority.",
        metadata={"source": "Facts"}
    )

    model_id = "all-MiniLM-L6-v2"
    model_path = "./llm/" + model_id
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_path)
    except:
        embeddings = HuggingFaceEmbeddings(model_name=model_id)
        # embeddings(model_path) -> save

    return Chroma.from_documents(
        [init_document],
        embedding=embeddings,
        persist_directory="./vectorstore"
    )

def get_model(model_id, train):
    model_path = "./llm/" + model_id
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except:
        if train:
            id2label = {0: Category.TASK.value, 1: Category.KNOWLEDGE.value, 2: Category.QUESTION.value}
            label2id = {Category.TASK.value: 0, Category.KNOWLEDGE.value: 1, Category.QUESTION.value: 2}
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=3, id2label=id2label, label2id=label2id
            )
            train_model(model, model_path, tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.save_pretrained(model_path)

    return tokenizer, model

def get_text_classification_pipeline():
    tokenizer, model = get_model("distilbert/distilbert-base-uncased", True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

def get_text_generation_pipeline():
    tokenizer, model = get_model("gpt2", False)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe)

def train_model(model, model_path, tokenizer):
    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        remove_unused_columns=False,
        #eval_strategy="no",
        save_strategy="epoch",
        #load_best_model_at_end=True,
        push_to_hub=False,
    )

    tokenized_dataset = load_dataset("json", data_files="categories.jsonl")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].class_encode_column("input"),
        #eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
        
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    init()
