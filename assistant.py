from dotenv import load_dotenv
import os 

from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

from agents.jira_task_agent import JiraTaskAgent

class PersonalAssistant:
    def __init__(self, llm: HuggingFacePipeline, user_email):
        load_dotenv()
        self.llm = llm

        # Setup agents
        self.jira_task_agent = JiraTaskAgent(os.environ['JIRA_URL'], os.environ['JIRA_TOKEN'], user_email)

    def test_prompt(self):
        text = "This is a great movie!"
        result = self.llm.invoke(text)
        print(result)

    def run(self):
        print("assitant running")
        self.test_prompt()
        # Fetch and prioritize Jira tasks
       # active_cards = self.jira_task_agent.get_active_cards()
       # prioritized_cards = self.jira_task_agent.analyze_and_prioritize_cards(active_cards)
       # print("Prioritized Jira Cards:", prioritized_cards)

        # Set reminders for upcoming tasks
       # for card in prioritized_cards:
       #     self.reminder_agent.set_reminder(card['fields']['summary'])

        # Check emails