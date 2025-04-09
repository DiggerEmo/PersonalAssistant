from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_chroma import Chroma

class TaskManagerAgent:
    def __init__(self, generationllm: HuggingFacePipeline, vectorstore: Chroma):
        self.generationllm = generationllm
        self.vectorstore = vectorstore

    def create_task(self, task: str):
        # create task
        print(task)

    def complete_task(self, task: str):
        # create task
        print(task)

    def get_next_task(self):
        # determine the task with the next highest priority and return it
        print("")

    def ask_question(self, question: str):
        prompt = self.build_prompt().format(input=question)
        print(self.generationllm.invoke(prompt))

    def build_prompt(self):
        # TODO this should be build from vector store?
        # or this should be more general, state important information for the users everyday work
        examples = [
            {
                "input": "I need to finish the project report by tomorrow.",
                "output": "Got it. I've added 'Finish the project report by tomorrow' to your task list."
            },
            {
                "input": "What tasks do I have for today?",
                "output": "Here are your tasks for today:\n1. Finish the High Prio Ticket\n2. Attend the team meeting at 9 AM\n3. Work on your AI Project"
            },
            {
                "input": "Please prioritize the Task 'Analyze PD issues for TA'",
                "output": "Sure, I've marked 'Analyze PD issues for TA' as a high prioity task and reprioritized other tasks accordingly.."
            },
        ]

        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Task: {input}\n{output}",
        )
        
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix="Task: {input}",
            input_variables=["input"]
        )