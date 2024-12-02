from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

class InformationAgent:
    def __init__(self, classificationllm: HuggingFacePipeline, vectorstore: Chroma):
        self.classificationllm = classificationllm
        self.vectorstore

    def add_info(self, info: str):
        # put info in DB
        print(info)

    def get_info(self, topic: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

        rag_chain = ({"knowledge": retriever, "copy": RunnablePassthrough()}
                    | self.build_prompt()
                    | self.classificationllm)
        response = rag_chain.invoke(topic)

        return response.content
    
    def build_prompt():
        message = """
        Retrieve information that is relavant to the following topic using the following knowledge:
        Knowledge:
        {knowledge}
        Topic:
        {copy}
        Fixed Topic:
        """
        return ChatPromptTemplate.from_messages([("human", message)])