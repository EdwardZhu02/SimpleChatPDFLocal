#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ====================================
# @File name        : app.py
# @First created    : 2024/8/12 11:45
# @Author           : Yuzhi Zhu
# @E-mail           : edwardmashed@gmail.com
# ====================================
import json
import os
import time
import secrets

from flask import Flask, render_template, request, redirect

import maininstance  # for ChatPDF app running

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded files
ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

session_global_dict = {}  # global variables for a single chat session, initialized when a session begin.


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_session_global_dict():
    global session_global_dict
    session_global_dict = {
        "chat_session_id": secrets.token_urlsafe(16),
        "IndvPDFInstance": maininstance.ChatPDFInstance(),
        "summary_text": "",
        "summary_questions": "",
        "current_question": "",
        "current_response": "",
        "chat_history_list": [],  # [[query1, response1], [query2, response2], ...]
    }


@app.route('/', methods=['GET'])
def home_page_handler():
    return redirect('/upload_file')


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file_handler():
    # POST request
    if request.method == 'POST':

        if 'file' in request.files:
            file = request.files['file']
            # If user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                raise FileNotFoundError

            if file and allowed_file(file.filename):  # valid file upload, begin chat

                initialize_session_global_dict()  # prepare session cache

                filename = file.filename
                fileid = session_global_dict["chat_session_id"] + "_file.pdf"
                filedir = os.path.join(app.config['UPLOAD_FOLDER'], fileid)
                file.save(filedir)  # no prob here, but pycharm kept throwing out warnings

                # Execute chatPDF with the uploaded file
                session_global_dict['IndvPDFInstance'] = maininstance.ChatPDFInstance(
                    filedir, filename, model_name='gemma2:2b'
                )
                session_global_dict['IndvPDFInstance'].split_embed_text()

                # print("Generating summary for abstract...\n")
                summary_text = None
                summary_questions = None
                try:
                    while True:
                        try:
                            summary_text = session_global_dict['IndvPDFInstance'].generate_summary_text()
                            summary_questions = \
                                session_global_dict['IndvPDFInstance'].generate_questions_based_on_summary(
                                    str(summary_text))
                            raise StopIteration
                        except json.decoder.JSONDecodeError:
                            # print("JSON decode error occurred. Retrying")
                            time.sleep(0.2)
                            continue
                except StopIteration:
                    pass
                # Save response to session
                session_global_dict['summary_text'] = summary_text
                session_global_dict['summary_questions'] = summary_questions

                return render_template(
                    'dialog_main.html',
                    file_name=session_global_dict['IndvPDFInstance'].show_file_names()[0],
                    file_display_name=session_global_dict['IndvPDFInstance'].show_file_names()[1],
                    summary_text=session_global_dict['summary_text'],
                    initial_question_text=session_global_dict['summary_questions'],
                )

    # GET request
    return render_template('upload_file.html')  # Render the initial page for file upload


@app.route('/dialog', methods=['POST'])
def dialog_handler():
    if request.form['question_query']:
        question_query = str(request.form['question_query'])

        # begin generating response to the question
        qachain_result = None
        try:
            # validate if instance is created by uploaded PDF
            if not session_global_dict['IndvPDFInstance'].show_file_names()[0]:
                raise StopIteration
            while True:
                try:
                    qachain_result = session_global_dict['IndvPDFInstance'].query_round_perform(question_query)
                    raise StopIteration
                except json.decoder.JSONDecodeError:
                    # print("JSON decode error occurred. Retrying")
                    time.sleep(0.2)
                    continue
        except StopIteration:
            pass

        session_global_dict['current_question'] = qachain_result['query']
        session_global_dict['current_response'] = qachain_result['result']
        session_global_dict['chat_history_list'].append({
            'query': qachain_result['query'],
            'result': qachain_result['result'],
        })

        chat_history_list_torender = session_global_dict['chat_history_list']
        # chat_history_list_torender.pop()  # remove the current conversation and render only the chat history

        print(chat_history_list_torender)  # debug

        return render_template(
            'dialog_main.html',
            show_dialog_panel=True,
            file_name=session_global_dict['IndvPDFInstance'].show_file_names()[0],
            file_display_name=session_global_dict['IndvPDFInstance'].show_file_names()[1],
            summary_text=session_global_dict['summary_text'],
            initial_question_text=session_global_dict['summary_questions'],
            chat_history_list=chat_history_list_torender,
            current_question=session_global_dict['current_question'],
            current_response=session_global_dict['current_response'],
        )


if __name__ == '__main__':
    app.run(debug=True)
