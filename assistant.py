from dotenv import load_dotenv
import os 

from langchain import LLMChain, PromptTemplate
from langchain_core.prompts import PromptTemplate

from agents.jira_task_agent import JiraTaskAgent

class PersonalAssistant:
    def __init__(self, llm, user_email):
        load_dotenv()
        self.llm = llm

        # Setup agents
        self.jira_task_agent = JiraTaskAgent(os.environ['JIRA_URL'], os.environ['JIRA_TOKEN'], user_email)

    def test_prompt(self):
        template = """
        Write me a poem about {topic}.
        """
        topic = "Machine Learning"

        prompt = PromptTemplate(input_variables=["topic"], template=template)

        chain = prompt | self.llm

        output = chain.invoke({"topic": topic})
        print(output)

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