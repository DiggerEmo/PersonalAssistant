from dotenv import load_dotenv
import os 
from categories import Category 

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_chroma import Chroma

from agents.jira_task_agent import JiraTaskAgent
from agents.information_agent import InformationAgent
from agents.task_manager_agent import TaskManagerAgent

class PersonalAssistant:
    def __init__(self, generate_llm: HuggingFacePipeline, classification_llm: HuggingFacePipeline, vectorstore: Chroma, user_email):
        load_dotenv()
        self.generate_llm = generate_llm
        self.classification_llm = classification_llm

        # Setup agents
        self.information_agent = InformationAgent(classification_llm, vectorstore)
        self.task_manager_agent = TaskManagerAgent(generate_llm, vectorstore)
        #self.jira_task_agent = JiraTaskAgent(os.environ['JIRA_URL'], os.environ['JIRA_TOKEN'], user_email)

    def add_task(self, task: str):
        self.task_manager_agent.create_task(task)

    def add_task(self, task: str):
        self.task_manager_agent.complete_task(task)

    def take_input(self, input: str):
        # Categorize the input into a known category
        category = self.categorize(input)

        if category == Category.TASK: 
            self.task_manager_agent.create_task(input)
        if category == Category.KNOWLEDGE:
            self.information_agent.add_info(input)
        if category == Category.QUESTION: 
            self.task_manager_agent.ask_question(input)        

    def categorize(self, input: str):
        self.classification_llm.invoke(input)

        return Category.TASK

    def get_next_task(self):
        print(self.task_manager_agent.get_next_task(input))