from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from llm_chatglm import ChatGLMLLM

memory = ConversationBufferWindowMemory(
    k=2, return_messages=True, output_key="result")


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGLMLLM()
    prompt_template = """
    你是一个专业的知识问答助手，根据提供的内容，简洁准确地回答用户的问题。
    如果内容不足以回答，请礼貌告知用户无法回答。

    上下文内容：
    {context}

    问题：
    {question}

    答案：
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory=memory
    )
    return qa_chain
