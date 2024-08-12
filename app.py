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

from flask import Flask, render_template, request, session

import maininstance  # for ChatPDF app running

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'dZs2nvojIxFQR3ynqfaO5eT1GBmQOzYVcKxaZMEgGto'  # secrets.token_urlsafe(32)

UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded files
ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global_dict = {
    "IndvPDFInstance": maininstance.ChatPDFInstance(),
    "summary_text": "",
    "summary_questions": "",
    "current_question": "",
    "current_response": "",
}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def chatpdf_handler():
    # POST request
    if request.method == 'POST':

        if 'file' in request.files:
            file = request.files['file']
            # If user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                raise FileNotFoundError

            if file and allowed_file(file.filename):
                filename = file.filename
                filedir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filedir)

                # Execute chatPDF with the uploaded file
                global_dict['IndvPDFInstance'] = maininstance.ChatPDFInstance(filedir)
                global_dict['IndvPDFInstance'].split_embed_text()

                print("Generating summary for abstract...\n")
                summary_text = None
                summary_questions = None
                try:
                    while True:
                        try:
                            summary_text = global_dict['IndvPDFInstance'].generate_summary_text()
                            summary_questions = global_dict['IndvPDFInstance'].generate_questions_based_on_summary(
                                str(summary_text))
                            raise StopIteration
                        except json.decoder.JSONDecodeError:
                            # print("JSON decode error occurred. Retrying")
                            time.sleep(0.2)
                            continue
                except StopIteration:
                    pass
                # Save response to session
                global_dict['summary_text'] = summary_text
                global_dict['summary_questions'] = summary_questions

                return render_template(
                    'index.html',
                    show_dialog_panel=True,
                    file_name=global_dict['IndvPDFInstance'].show_file_name(),
                    summary_text=global_dict['summary_text'],
                    initial_question_text=global_dict['summary_questions'],
                )

    # GET request
    return render_template('index.html', show_dialog_panel=False)  # Render the initial page


@app.route('/dialog', methods=['POST'])
def dialog_handler():
    if request.form['question_query']:
        question_query = str(request.form['question_query'])

        # begin generating response to the question
        qachain_result = None
        try:
            # validate if instance is created by uploaded PDF
            if not global_dict['IndvPDFInstance'].show_file_name():
                raise StopIteration
            while True:
                try:
                    qachain_result = global_dict['IndvPDFInstance'].query_round_perform(question_query)
                    raise StopIteration
                except json.decoder.JSONDecodeError:
                    # print("JSON decode error occurred. Retrying")
                    time.sleep(0.2)
                    continue
        except StopIteration:
            pass

        global_dict['current_question'] = question_query
        global_dict['current_response'] = qachain_result

        return render_template(
            'index.html',
            show_dialog_panel=True,
            file_name=global_dict['IndvPDFInstance'].show_file_name(),
            summary_text=global_dict['summary_text'],
            initial_question_text=global_dict['summary_questions'],
            current_question=global_dict['current_question'],
            current_response=global_dict['current_response'],
        )


if __name__ == '__main__':
    app.run(debug=True)
