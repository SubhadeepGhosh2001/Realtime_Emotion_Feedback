from flask import Flask, request, render_template, redirect, session, Response, url_for,jsonify
from db import __get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import time
import numpy as np
from keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import requests



# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("App_Secret_Key")

# Configure genai with the API key from .env
genai.configure(api_key=os.getenv("Gemini_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('model/model.h5')  # Ensure this path is correct
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def greet():
    return render_template("home.html",user_name=session.get('user_name'))

from flask import Flask, request, jsonify, session, redirect
from werkzeug.security import generate_password_hash

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle JSON or form data
        if request.is_json:
            data = request.get_json()
            name = data.get('name')
            email = data.get('email')
            raw_password = data.get('password')
            confirm_password = data.get('confirm_password')
        else:
            name = request.form.get('name')
            email = request.form.get('email')
            raw_password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')

        # Validate input
        if not name or not email or not raw_password or not confirm_password:
            return jsonify({'error': 'missing_fields', 'message': 'All fields are required.'}), 400

        if raw_password != confirm_password:
            return jsonify({'error': 'password_mismatch', 'message': 'Passwords do not match.'}), 400

        # Check if email exists
        conn = __get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM auth_table WHERE LOWER(email) = LOWER(%s)", (email,))
        user = cursor.fetchone()

        if user:
            cursor.close()
            conn.close()
            return jsonify({'error': 'email_exists', 'message': 'Email already registered. Please use a different email.'}), 400

        # Hash password and insert new user
        password = generate_password_hash(raw_password)
        try:
            cursor.execute(
                "INSERT INTO auth_table (name, email, password) VALUES (%s, %s, %s)",
                (name, email, password)
            )
            conn.commit()

            # Optionally set session data (if you want to auto-login after registration)
            cursor.execute("SELECT * FROM auth_table WHERE LOWER(email) = LOWER(%s)", (email,))
            new_user = cursor.fetchone()
            session['user_id'] = new_user['id']
            session['user_name'] = new_user['name']
            session['user_email'] = new_user['email']
            session['logged_in'] = True
            session['role'] = new_user.get('role', 'user')  # Default to 'user' if role isn't in table
        except Exception as e:
            conn.rollback()
            cursor.close()
            conn.close()
            return jsonify({'error': 'database_error', 'message': 'An error occurred during registration.'}), 500
        finally:
            cursor.close()
            conn.close()

        return jsonify({'message': 'Registration successful', 'redirect': '/home'}), 200

    # For GET requests, render the registration page
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check if request contains JSON or form data
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            email = request.form.get('email')
            password = request.form.get('password')

        if not email or not password:
            return jsonify({'error': 'missing_fields', 'message': 'All fields are required.'}), 400

        conn = __get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM auth_table WHERE LOWER(email) = LOWER(%s)", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return jsonify({'error': 'user_not_found', 'message': 'Account not registered. Please create an account.'}), 401

        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'invalid_password', 'message': 'Invalid password.'}), 401

        # Successful login
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        session['logged_in'] = True
        session['role'] = user['role']
        return jsonify({'message': 'Login successful', 'redirect': '/home'}), 200

    # For GET requests, render the login page
    return render_template('login.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # Check if request contains JSON or form data
#         if request.is_json:
#             data = request.get_json()
#             email = data.get('email')
#             password = data.get('password')
#             recaptcha_response = data.get('g-recaptcha-response')
#         else:
#             email = request.form.get('email')
#             password = request.form.get('password')
#             recaptcha_response = request.form.get('g-recaptcha-response')

#         if not email or not password or not recaptcha_response:
#             return jsonify({'error': 'missing_fields', 'message': 'All fields are required, including reCAPTCHA.'}), 400

#         # Verify reCAPTCHA
#         recaptcha_secret = '6LcqJGYrAAAAAEA8siYvrQzPxJrpmBXWsDfx_87R'  # Secret Key for server-side verification
#         recaptcha_verify_url = 'https://www.google.com/recaptcha/api/siteverify'
#         recaptcha_data = {
#             'secret': recaptcha_secret,
#             'response': recaptcha_response,
#             'remoteip': request.remote_addr
#         }
#         recaptcha_response = requests.post(recaptcha_verify_url, data=recaptcha_data)
#         recaptcha_result = recaptcha_response.json()

#         if not recaptcha_result.get('success'):
#             return jsonify({'error': 'recaptcha_failed', 'message': 'reCAPTCHA verification failed. Please try again.'}), 400

#         conn = __get_db_connection()
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute("SELECT * FROM auth_table WHERE LOWER(email) = LOWER(%s)", (email,))
#         user = cursor.fetchone()
#         cursor.close()
#         conn.close()

#         if not user:
#             return jsonify({'error': 'user_not_found', 'message': 'Account not registered. Please create an account.'}), 401

#         if not check_password_hash(user['password'], password):
#             return jsonify({'error': 'invalid_password', 'message': 'Invalid password.'}), 401

#         # Successful login
#         session['user_id'] = user['id']
#         session['user_name'] = user['name']
#         session['user_email'] = user['email']
#         session['logged_in'] = True
#         session['role'] = user['role']
#         return jsonify({'message': 'Login successful', 'redirect': '/home'}), 200

#     # For GET requests, render the login page
#     return render_template('login.html', site_key='6LcqJGYrAAAAAI5Xki-sm2j8Q8hCh58Ah2EPxKiu')  # Site Key for client-side widget

@app.route('/home')
def home():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login') 
    conn = __get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM auth_table")
    register_form = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('home.html', register_form=register_form, user_name=session.get('user_name'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('logged_in', None)
    return redirect('/login')

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect('/login')
    return Response(generate_frames(session['user_id']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(user_id):
    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        if time.time() - start_time > 8:
            break

        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]
            confidence = float(round(np.max(prediction) * 100, 2))


            # Save to DB (move this from outside request context)
            try:
                conn = __get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO emotion_logs (user_id, emotion, confidence) VALUES (%s, %s, %s)",
                    (user_id, label, confidence)
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print("DB Insert Error:", e)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        frame = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()


@app.route('/stream')
def stream():
    return render_template("video_feed.html",user_name=session.get('user_name'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login')
    user_id = session['user_id']
    conn = __get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
    SELECT emotion, COUNT(*) as count
    FROM emotion_logs
    WHERE user_id = %s
    GROUP BY emotion
    ORDER BY count DESC
""", (user_id,))

    emotion_data = cursor.fetchall()
    cursor.execute("""
        SELECT DATE(timestamp) as date, emotion, COUNT(*) as count
        FROM emotion_logs
        WHERE user_id = %s
        GROUP BY DATE(timestamp), emotion
        ORDER BY date
    """, (user_id,))
    timeline_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("dashboard.html", emotion_data=emotion_data, timeline_data=timeline_data,user_name=session.get('user_name'))

@app.route('/feedback')
def feedback():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login')

    user_id = session['user_id']
    
    # Fetch recent emotion stats from DB
    conn = __get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT emotion, COUNT(*) as count
        FROM emotion_logs
        WHERE user_id = %s
        GROUP BY emotion
    """, (user_id,))
    emotion_data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Prepare a text summary for Gemini
    summary = "User's recent emotional summary:\n"
    for row in emotion_data:
        summary += f"{row['emotion']}: {row['count']} times\n"

    # Prepare Gemini prompt
    prompt = (
        "You are an AI emotional coach. "
        "Based on the following emotional summary, provide constructive and friendly feedback to the user "
        "to help them improve their mental well-being:\n\n"
        f"{summary}\n\n"
        "Please keep it positive and supportive."
    )

    # Generate feedback using Gemini
    try:
        response = model.generate_content(prompt)
        feedback_text = response.text
    except Exception as e:
        feedback_text = f"Error generating feedback: {e}"

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d | %H:%M:%S")
    # Render feedback page
    return render_template("feedback.html", feedback=feedback_text, user_name=session.get('user_name'), current_time=current_time)



if __name__ == '__main__':
    app.run(debug=True, port=5000)









