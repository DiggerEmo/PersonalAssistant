from dotenv import load_dotenv
import os 

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_chroma import Chroma

from agents.jira_task_agent import JiraTaskAgent
from agents.information_agent import InformationAgent

class PersonalAssistant:
    def __init__(self, generate_llm: HuggingFacePipeline, classification_llm: HuggingFacePipeline, vectorstore: Chroma, user_email):
        load_dotenv()
        self.generate_llm = generate_llm
        self.classification_llm = classification_llm

        # Setup agents
        self.information_agent = InformationAgent(classification_llm, vectorstore)
        self.jira_task_agent = JiraTaskAgent(os.environ['JIRA_URL'], os.environ['JIRA_TOKEN'], user_email)

    def test_prompt(self):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are soto zen master Roshi."),
                ("human", "What is the essence of Zen?"),
                ("ai", "When you are hungry, eat. When you are tired, sleep."),
                ("human", "Respond to the question: {question}")
            ]
        )
        
        llm_chain = prompt_template | self.generate_llm
        result = llm_chain.invoke({"question": "What is the meaning of life?"})
        print(result)

    def run(self):
        print("assitant running")
        #self.test_prompt()
        self.information_agent.get_info("What application is most important for my work?")

        # Fetch and prioritize Jira tasks
       # active_cards = self.jira_task_agent.get_active_cards()
       # prioritized_cards = self.jira_task_agent.analyze_and_prioritize_cards(active_cards)
       # print("Prioritized Jira Cards:", prioritized_cards)

        # Set reminders for upcoming tasks
       # for card in prioritized_cards:
       #     self.reminder_agent.set_reminder(card['fields']['summary'])

        # Check emails