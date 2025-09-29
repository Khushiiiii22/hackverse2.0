from flask import Flask, request, jsonify, send_from_directory, render_template, send_file, session
import pandas as pd
from io import BytesIO
from burnout_prevention import analyze_burnout
from rag.rag import simplify_legal_doc, load_and_split, build_chain, vectorstore
from news_myth import analyze_audit
from data_safeguard import run_audit, apply_corrections
import os
import threading
import uuid

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

# Configure session secret key
secret = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
if not secret:
    secret = os.urandom(24).hex()
    print("Warning: FLASK_SECRET_KEY not set. Using ephemeral secret key for development.")
app.secret_key = secret

last_burnout_result = {"detected_issues": [], "intervention_message": ""}
burnout_cond = threading.Condition()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/burnoutprevention', methods=['GET'])
def burnoutprevention_form():
    print("Serving burnout results page")
    return send_from_directory(app.static_folder, 'burnout_results.html')



@app.route('/api/burnout', methods=['POST'])
def api_burnout():
    session_data = request.json or {}
    global last_burnout_result
    try:
        result = analyze_burnout(session_data)
        with burnout_cond:
            last_burnout_result = result
            burnout_cond.notify_all()
        return jsonify(result)
    except Exception as e:
        print("Burnout analysis error:", e)
        return jsonify({"error": "analysis failed"}), 500


@app.route('/api/burnout/latest', methods=['GET'])
def api_burnout_latest():
    global last_burnout_result
    resp = jsonify(last_burnout_result)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/api/rag', methods=['POST'])
def api_rag():
    doc_data = request.json
    result = simplify_legal_doc(doc_data)
    return jsonify(result)


@app.route('/api/audit', methods=['POST'])
def api_audit():
    data = request.json
    result = analyze_audit(data)
    return jsonify(result)


@app.route('/datasafeguard', methods=['GET'])
def datasafeguard_form():
    return render_template('datasafeguard.html')


@app.route('/datasafeguard', methods=['POST'])
def datasafeguard_process():
    if 'dataset' not in request.files:
        return "No file uploaded", 400
    file = request.files['dataset']
    df = pd.read_csv(file)
    orig_score, orig_insights = run_audit(df, file.filename)
    corrected_df = apply_corrections(df)
    trust_score, insights = run_audit(corrected_df, file.filename)
    original_table = df.head(10).to_html(classes='table table-striped', index=False)
    corrected_table = corrected_df.head(10).to_html(classes='table table-striped', index=False)
    csv_buffer = BytesIO()
    corrected_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    app.config['corrected_csv'] = csv_buffer
    return render_template(
        'datasafeguard_result.html',
        trust_score=trust_score,
        orig_score=orig_score,
        insights=insights,
        original_table=original_table,
        corrected_table=corrected_table
    )


@app.route('/datasafeguard/download')
def datasafeguard_download():
    csv_buffer = app.config.get('corrected_csv')
    if not csv_buffer:
        return "No file ready for download", 404
    csv_buffer.seek(0)
    return send_file(
        csv_buffer,
        as_attachment=True,
        download_name="corrected_dataset.csv",
        mimetype="text/csv"
    )


@app.route('/newsmyth', methods=['GET'])
def newsmyth_form():
    return render_template('news_verify.html')


@app.route('/newsverify', methods=['POST'])
def news_verify_process():
    claim = request.form.get('claim_text', '').strip()
    if not claim:
        return render_template('news_verify.html', error="Please enter a news claim.")
    try:
        result = analyze_audit(claim)
        print("Analysis result:", result)
        return render_template('news_verify_results.html', result=result)
    except Exception as e:
        print("Error in analysis:", str(e))
        return render_template('news_verify.html', error="An error occurred during analysis. Please try again.")


@app.route('/rag', methods=['GET', 'POST'])
def rag_page():
    if request.method == 'POST':
        uploaded_file = request.files.get('docfile')
        if not uploaded_file:
            return render_template('rag_upload.html', error="Please upload a document file.")

        file_ext = os.path.splitext(uploaded_file.filename)[1]
        tmp_filename = f"{uuid.uuid4()}{file_ext}"
        tmp_filepath = os.path.join("/tmp", tmp_filename)
        uploaded_file.save(tmp_filepath)

        from rag.custom_splitter import load_and_split
        from rag.vectorstore import vectorstore

        docs = load_and_split(tmp_filepath)

        for doc in docs:
            doc.metadata.update({"doc_id": tmp_filename, "name": uploaded_file.filename})

        vectorstore.add_documents(docs)

        session['rag_doc_id'] = tmp_filename

        os.remove(tmp_filepath)
        summary_text = "Document processed successfully!"
        fulltext = "\n\n".join(doc.page_content for doc in docs)
        return render_template('rag_results.html', summary_text=summary_text, fulltext=fulltext)

    return render_template('rag_upload.html')


@app.route('/api/rag/qna', methods=['POST'])
def rag_qna():
    question = request.json.get('question')
    doc_id = session.get('rag_doc_id')
    if not doc_id:
        return jsonify({"answer": "Please upload a document first."})

    from rag.chains import build_chain
    chain = build_chain(doc_id)
    response = chain.invoke({"input": question})
    answer = response.get("answer", "Sorry, no answer available.")
    return jsonify({"answer": answer})


@app.route('/js/<path:filename>')
def frontend_js(filename):
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)


@app.route('/<path:filename>')
def frontend_static(filename):
    allowed = {'styles.css', 'app.js'}
    if filename in allowed:
        return send_from_directory(app.static_folder, filename)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
