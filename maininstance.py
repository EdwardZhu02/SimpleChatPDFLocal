#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ====================================
# @File name        : maininstance.py
# @First created    : 2024/8/11 14:31
# @Author           : Yuzhi Zhu
# @E-mail           : edwardmashed@gmail.com
# ====================================

from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text split
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader, PyPDFLoader
import sys
import os
import json
import time


class SuppressStdout:  # define class to supress verbose output
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class ChatPDFInstance:
    def __init__(self, file_path=None, model_name='llama3.1'):
        self._vectorstore = None
        self._all_splits = None
        if file_path:
            self._file_path = file_path
            self._file_name = file_path.split("/")[-1]
            self._llm = Ollama(model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            # Load text from original file
            loader = self._file_path.endswith(".pdf") and PyPDFLoader(self._file_path) or TextLoader(self._file_path)
            self._textraw = loader.load()
        else:
            self._file_name = None  # blank object, serve as identifier

    def split_embed_text(self):
        print("Splitting text and creating embedding database...")
        with SuppressStdout():
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            self._all_splits = text_splitter.split_documents(self._textraw)
            # Create temporary embedding database and return
            self._vectorstore = Chroma.from_documents(documents=self._all_splits, embedding=GPT4AllEmbeddings())

    def generate_summary_text(self):
        prompt_template = """You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.
        Write a short summary of the following text. Don't output anything except the summary itself.
        Text: "{text}"
        Concise Summary:"""
        SUMMARY_CHAIN_PROMPT = PromptTemplate.from_template(template=prompt_template)
        summary_chain = load_summarize_chain(self._llm, chain_type="stuff", prompt=SUMMARY_CHAIN_PROMPT, verbose=False)
        _summary_chain_result = summary_chain.run(self._all_splits[:8])  # Abstract-only summary
        return _summary_chain_result

    def generate_questions_based_on_summary(self, summary_chain_result):
        prompt_template = """You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.
        Given the following text summary, generate three relevant questions that help users to further comprehend the text.
        Keep the questions as concise as possible, don't output anything except the questions.
        Summary: {text}
        Questions:"""

        SUMMARY_QUESTIONS_PROMPT = PromptTemplate(input_variables=["text"], template=prompt_template)
        summary_questions_chain = LLMChain(llm=self._llm, prompt=SUMMARY_QUESTIONS_PROMPT)
        _summary_questions_chain_result = summary_questions_chain.run(summary_chain_result)
        return _summary_questions_chain_result

    def query_round_perform(self, query_text):
        prompt_template = """You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Context: "{context}"
        Question: {question}
        Helpful Answer:"""

        # Create final prompt
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Create QA chain with model and prompt
        # Reference for obtaining the complete prompt:
        # https://stackoverflow.com/questions/77352474/langchain-how-to-get-complete-prompt-retrievalqa-from-chain-type
        qa_chain = RetrievalQA.from_chain_type(
            self._llm,
            retriever=self._vectorstore.as_retriever(search_kwargs={"k": 8, "score_threshold": 0.01}),
            chain_type_kwargs={"verbose": False, "prompt": QA_CHAIN_PROMPT},
        )

        # Get result output and return
        _qachain_result = qa_chain({"query": str(query_text)})
        return _qachain_result

    def show_all_splits(self):
        return [i.page_content for i in self._all_splits]

    def show_file_name(self):
        return self._file_name


if __name__ == "__main__":  # Launch app in CLI
    file_dir = input("""
Specify PDF or text file to load.
Please enclose the file directory in single brackets.
Example: \'/path/to/file\':""")
    if file_dir.startswith('\'') and file_dir.endswith('\''):
        file_dir = file_dir.rstrip('\'').lstrip('\'')

    file_name = file_dir.split("/")[-1]
    IndvPDFInstance = ChatPDFInstance(file_dir)
    IndvPDFInstance.split_embed_text()

    try:
        print("Generating summary for abstract...\n")
        while True:
            try:
                summary_text = IndvPDFInstance.generate_summary_text()
                print("\n\n")
                summary_questions = IndvPDFInstance.generate_questions_based_on_summary(str(summary_text))
                raise StopIteration
            except json.decoder.JSONDecodeError:
                # print("JSON decode error occurred. Retrying")
                time.sleep(0.2)
                summary_text = None
                summary_questions = None
                continue
    except StopIteration:
        summary_text = None
        summary_questions = None
        pass

    while True:
        query = input("\n\n=====\n[%s]\nQuery (/ for cmds, /h for help): " % file_name)
        if query in ["/exit", "/quit", "/q"]:
            break
        elif query in ["/showsummary", "/ss"]:
            print("Contexts used:", "\n\n".join(IndvPDFInstance.show_all_splits()[:10]),
                  "=====", "Summary:", summary_text, sep="\n")
            continue
        elif query in ["/showquestions", "/sq"]:
            print(summary_questions)
            continue
        elif query == "/h":
            print("""
Available commands:
- /exit, /quit, /q : quit the conversation;
- /showsummary, /ss : show manuscript summary as well as original text used for generation;
- /showquestions, /sq : show example questions generated using manuscript summary.
            """)
            continue
        if query.strip() == "":
            continue

        try:
            while True:
                try:
                    qachain_result = IndvPDFInstance.query_round_perform(query)
                    raise StopIteration
                except json.decoder.JSONDecodeError:
                    # print("JSON decode error occurred. Retrying")
                    time.sleep(0.2)
                    continue
        except StopIteration:
            pass
