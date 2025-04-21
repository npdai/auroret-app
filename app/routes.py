from flask import Blueprint, render_template, request, redirect, url_for, flash, session # type: ignore
from app.database import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from PIL import Image # type: ignore
import os
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import io
from fpdf import FPDF # type: ignore
from flask import send_file # type: ignore
import requests
import zipfile
import gdown





main = Blueprint('main', __name__)

MODEL_PATH = 'ce_45_DR-DME_model'

def download_model_from_drive():
    print("🟡 Model not found. Downloading from Google Drive...")

    # Your new file ID
    file_id = "19zF2IVTMhnVScgoSWFnH9e67eUqlevzJ"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    zip_path = "model.zip"

    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(zip_path)
        print("✅ Model downloaded and extracted successfully.")

    except Exception as e:
        print("❌ Model download failed:", e)

# Trigger download only if model folder doesn't exist
if not os.path.exists(MODEL_PATH):
    download_model_from_drive()

# Load the model
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]


def resize_fundus_image(input_path, output_path, size=(779, 779)):
    image = Image.open(input_path).convert('RGB')
    resized_image = image.resize(size, Image.LANCZOS)
    resized_image.save(output_path)


def image_to_byte_string(image_array):
    img_bytes = io.BytesIO()
    image = Image.fromarray(image_array)
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


def predict(image_array, infer):
    image_string = image_to_byte_string(image_array)
    inputs = {'input_image': tf.constant([image_string])}
    return infer(**inputs)


def get_result(data):
    DME = ['DME_NO', 'DME_YES']
    DR = ['DR_NO', 'DR_MILD', 'DR_MODERATE', 'DR_SEVERE', 'DR_PROLIFERATIVE']

    DME_result = data["OUTPUT_DME_INDEX_SELECTED_VIA_CASCADING_THRESHOLDS"].numpy()[0]
    DR_result = data["OUTPUT_DR_GRADE_INDEX_SELECTED_VIA_CASCADING_THRESHOLDS"].numpy()[0]
    DME_gradability = data["OUTPUT_DME_GRADABILITY_INDEX_SELECTED_VIA_CASCADING_THRESHOLDS"].numpy()[0]
    DR_gradability = data["OUTPUT_DR_GRADABILITY_INDEX_SELECTED_VIA_CASCADING_THRESHOLDS"].numpy()[0]

    return {
        'DR': DR[int(DR_result)] if DR_gradability == 0 else "DR_UNGRADABLE",
        'DME': DME[int(DME_result)] if DME_gradability == 0 else "DME_UNGRADABLE"
    }

def get_diagnostic_summary(dr, dme):
    summary = {
        ('DR_UNGRADABLE', 'DME_UNGRADABLE'): "The fundus image is ungradable for both Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME). A repeat scan or manual clinical assessment is recommended.",
        ('DR_UNGRADABLE', 'DME_NO'): "DR diagnosis is ungradable due to image quality. No signs of Diabetic Macular Edema (DME-negative) detected.",
        ('DR_UNGRADABLE', 'DME_YES'): "DR diagnosis is ungradable. Presence of Diabetic Macular Edema (DME-positive) observed. Immediate ophthalmic consultation advised.",
        
        ('DR_NO', 'DME_UNGRADABLE'): "No signs of Diabetic Retinopathy (DR-negative). DME status is ungradable due to image quality. Further examination is suggested.",
        ('DR_NO', 'DME_NO'): "Patient exhibits no signs of Diabetic Retinopathy (DR-negative) or Diabetic Macular Edema (DME-negative).",
        ('DR_NO', 'DME_YES'): "Patient exhibits no signs of Diabetic Retinopathy (DR-negative). Evidence of Diabetic Macular Edema (DME-positive) is present. Recommended: Immediate ophthalmic evaluation.",

        ('DR_MILD', 'DME_UNGRADABLE'): "Mild signs of Diabetic Retinopathy detected. DME status is ungradable. Recommend closer monitoring and follow-up imaging.",
        ('DR_MILD', 'DME_NO'): "Mild non-proliferative Diabetic Retinopathy detected. No DME present. Suggest monitoring and lifestyle changes.",
        ('DR_MILD', 'DME_YES'): "Mild DR with co-existing Diabetic Macular Edema (DME-positive). Early-stage intervention is advised.",

        ('DR_MODERATE', 'DME_UNGRADABLE'): "Moderate DR identified. DME status is ungradable. Suggest further evaluation to assess macular involvement.",
        ('DR_MODERATE', 'DME_NO'): "Moderate DR with no signs of DME. Recommend regular follow-up and glycemic control.",
        ('DR_MODERATE', 'DME_YES'): "Moderate DR and presence of DME detected. Treatment planning should begin with a retinal specialist.",

        ('DR_SEVERE', 'DME_UNGRADABLE'): "Severe DR identified. DME ungradable. Urgent follow-up recommended to assess for potential edema.",
        ('DR_SEVERE', 'DME_NO'): "Severe DR noted. No DME detected. Urgent care and retinal monitoring essential.",
        ('DR_SEVERE', 'DME_YES'): "Severe DR with co-existing DME. Immediate intervention and specialist care is highly recommended.",

        ('DR_PROLIFERATIVE', 'DME_UNGRADABLE'): "Proliferative DR detected. DME ungradable. Urgent attention is required to assess full disease extent.",
        ('DR_PROLIFERATIVE', 'DME_NO'): "Proliferative DR present. No signs of DME. Patient requires laser or surgical management.",
        ('DR_PROLIFERATIVE', 'DME_YES'): "Proliferative DR and DME confirmed. Critical case requiring urgent retinal intervention and treatment."
    }

    return summary.get((dr, dme), "Diagnostic information unavailable. Please verify input or consult a specialist.")



@main.route('/')
def index():
    return render_template('index.html')


@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        phone = request.form['phone']
        medical_reg_no = request.form['medical_reg_no']
        hospital = request.form['hospital']

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('main.register'))

        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO doctors (full_name, email, password, phone, medical_reg_no, hospital)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (full_name, email, generate_password_hash(password), phone, medical_reg_no, hospital))
            conn.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for('main.index'))
        except Exception as e:
            print("ERROR:", e)
            flash("Account creation failed. Email might already exist.", "error")
        finally:
            conn.close()

    return render_template('register.html')


@main.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    doctor = conn.execute('SELECT * FROM doctors WHERE email = ?', (email,)).fetchone()
    conn.close()

    if doctor and check_password_hash(doctor['password'], password):
        session['doctor_id'] = doctor['id']
        session['doctor_name'] = doctor['full_name']
        session.modified = True  # Ensure session changes are committed
        flash("Login successful!", "success")
        return redirect(url_for('main.home'))  # Redirect to home after login
    else:
        flash("Invalid email or password.", "error")
        return redirect(url_for('main.index'))



@main.route('/logout')
def logout():
    session.pop('doctor_id', None)
    session.pop('doctor_name', None)
    session.clear()  # Ensure session is completely cleared
    flash("You have been logged out.", "info")
    return redirect(url_for('main.index'))


@main.route('/home')
def home():
    if 'doctor_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('main.index'))
    return render_template('home.html', doctor_name=session['doctor_name'])

# The rest of your routes remain unchanged.



@main.route('/image-analysis', methods=['GET', 'POST'])
def image_analysis():
    result = None
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        eye_side = request.form['OD_OS']
        image = request.files['image']

        UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            original_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(original_path)

            preprocessed_filename = 'preprocessed_' + filename
            preprocessed_path = os.path.join(UPLOAD_FOLDER, preprocessed_filename)
            image_path_for_db = f'static/uploads/{preprocessed_filename}'

            try:
                # Resize
                resize_fundus_image(original_path, preprocessed_path)

                # Predict
                image_array = np.array(Image.open(preprocessed_path).convert('RGB'))
                raw_prediction = predict(image_array, infer)
                prediction_result = get_result(raw_prediction)

                # Prepare result dict with image path and patient ID
                result = {
                    "DR": prediction_result["DR"],
                    "DME": prediction_result["DME"],
                    "image_path": "/" + image_path_for_db,  # For browser access
                    "patient_id": patient_id
                }

                # Save to DB with eye_side update
                conn = get_db_connection()
                conn.execute('''
                    UPDATE patients
                    SET image_path = ?, results = ?, eye_side = ?
                    WHERE patient_id = ?
                ''', (image_path_for_db, f"DR: {result['DR']}, DME: {result['DME']}", eye_side, patient_id))
                conn.commit()
                conn.close()

                flash("Image processed and prediction successful!", "success")

            except Exception as e:
                print("PREDICTION ERROR:", e)
                flash("AI model prediction failed.", "error")
        else:
            flash("Invalid file format. Please upload JPG or PNG.", "error")

    return render_template('image-analysis.html', result=result)




@main.route('/register-patient', methods=['GET', 'POST'])
def patient_register():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        dob = request.form['dob']
        age = request.form['age']
        patient_id = request.form['patient_id']
        contact = request.form['contact']
        visit_date = request.form['visit_date']
        medical_history = request.form['medical_history']

        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO patients (name, gender, dob, age, patient_id, contact, visit_date, medical_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, gender, dob, age, patient_id, contact, visit_date, medical_history))
            conn.commit()
            flash("Patient registered successfully!", "success")
            return redirect(url_for('main.home'))
        except Exception as e:
            print("PATIENT REGISTRATION ERROR:", e)
            flash("Failed to register patient. Patient ID might already exist.", "error")
        finally:
            conn.close()

    return render_template('patient-registration.html')

@main.route('/search-patient', methods=['GET', 'POST'])
def patient_search():
    patients = []
    if request.method == 'POST':
        search_query = request.form['search_query']
        conn = get_db_connection()
        patients = conn.execute('''
            SELECT * FROM patients 
            WHERE name LIKE ? OR patient_id LIKE ?
        ''', (f'%{search_query}%', f'%{search_query}%')).fetchall()
        conn.close()

    return render_template('search-patient.html', patients=patients)

@main.route('/save-result', methods=['POST'])
def save_result():
    patient_id = request.form['patient_id']
    image_path = request.form['image_path']
    dr = request.form['dr']
    dme = request.form['dme']
    result_text = f"DR: {dr}, DME: {dme}"

    conn = get_db_connection()
    try:
        conn.execute('''
            UPDATE patients
            SET image_path = ?, results = ?
            WHERE patient_id = ?
        ''', (image_path, result_text, patient_id))
        conn.commit()
        flash("Prediction result saved successfully!", "success")

        
        result = {
            'patient_id': patient_id,
            'DR': dr,
            'DME': dme,
            'image_path': image_path
        }
        return render_template('image-analysis.html', result=result)

    except Exception as e:
        print("SAVE ERROR:", e)
        flash("Failed to save results to database.", "error")
        return redirect(url_for('main.image_analysis'))
    finally:
        conn.close()



@main.route('/download-report', methods=['POST'])
def download_report():
    from fpdf import FPDF # type: ignore
    import datetime

    patient_id = request.form['patient_id']

    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
    conn.close()

    if not patient:
        flash("Patient not found!", "error")
        return redirect(url_for('main.image_analysis'))

    # Setup PDF
    pdf = FPDF()
    pdf.add_page()

    # Add top logo
    top_logo_path = os.path.join("app", "static", "image", "logoo.png")
    if os.path.exists(top_logo_path):
        pdf.image(top_logo_path, x=10, y=5, w=190)

    pdf.set_xy(10, 45)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Report", ln=True, align="C")

    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}", ln=True, align="C")
    pdf.ln(10)

    def row(label, value):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(50, 8, f"{label}:", border=0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, str(value), ln=True)

    row("Patient Name", patient['name'])
    row("Patient ID", patient['patient_id'])
    row("Age", patient['age'])
    row("Gender", patient['gender'])
    row("DOB", patient['dob'])
    row("Contact", patient['contact'])
    row("Visit Date", patient['visit_date'])

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Medical History", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, patient['medical_history'] or "NA")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "AI Prediction Result", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, patient['results'] or "NA")

    # Retina Image (centered)
    if patient['image_path']:
        image_path = os.path.join(os.getcwd(), "app", patient['image_path'])
        if os.path.exists(image_path):
            pdf.ln(5)
            x_center = (210 - 100) / 2
            pdf.image(image_path, x=x_center, w=100)

    # Add footer image
    footer_path = os.path.join(os.getcwd(), "app", "static", "image", "footer.png")
    if os.path.exists(footer_path):
        pdf.image(footer_path, x=0, y=277, w=210)

    # Output path (absolute & correct)
    reports_dir = os.path.join(os.getcwd(), "app", "static", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"{patient_id}_report.pdf")
    pdf.output(pdf_path)

    # Confirm file exists before sending
    if not os.path.exists(pdf_path):
        flash("Report generation failed: file missing!", "error")
        return redirect(url_for('main.image_analysis'))

    return send_file(pdf_path, as_attachment=True)


@main.route('/preview-report/<patient_id>')
def preview_report(patient_id):
    if 'doctor_id' not in session:
        flash("Please log in to view the report.", "error")
        return redirect(url_for('main.index'))

    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
    doctor = conn.execute('SELECT * FROM doctors WHERE id = ?', (session['doctor_id'],)).fetchone()
    conn.close()

    if not patient or not doctor:
        flash("Patient or doctor not found.", "error")
        return redirect(url_for('main.home'))

    # Extract DR and DME from results string
    result_text = patient['results']
    try:
        dr = result_text.split("DR:")[1].split(",")[0].strip()
        dme = result_text.split("DME:")[1].strip()
    except Exception as e:
        dr, dme = "UNKNOWN", "UNKNOWN"

    # Get diagnostic summary
    diagnostic_summary = get_diagnostic_summary(dr, dme)

    return render_template(
        "preview-report.html",
        patient=patient,
        doctor=doctor,
        diagnostic_summary=diagnostic_summary
    )
