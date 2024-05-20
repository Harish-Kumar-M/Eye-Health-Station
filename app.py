import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
import base64
import pdfkit
from streamlit import session_state
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
import shutil
from tensorflow import keras
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

selected_model = "EfficientNetB0"
model = keras.models.load_model("models\\EfficientNetB0_model.h5")
hide_streamlit_style = """
                <style>
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
def signup(json_file_path="data.json"):
    form_styles = """
    <style>
    /* Customize text inputs */
    .stTextInput input {
        background-color: #f1f1f1;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus {
        outline: none;
        border-color: #4CAF50;
        box-shadow: 0 0 5px #4CAF50;
    }
    /* Customize radio buttons */
    .stRadio > div > div {
        background-color: #f1f1f1;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .stRadio > div > div:hover {
        background-color: #e0e0e0;
    }
    /* Customize form submit button */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 10px;
        cursor: pointer;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }
    /* Customize form submit button on hover */
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Customize form title */
    .stTitle {
        color: #4CAF50;
        font-size: 32px;
        margin-bottom: 30px;
    }
    </style>
    """
    st.markdown(form_styles, unsafe_allow_html=True)
    
    with st.form("signup_form"):
        st.markdown("<div class='stTitle'>Create an Account</div>", unsafe_allow_html=True)
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def predict(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    classes = ['Age Degeneration', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Others']
    return classes[np.argmax(predictions)]

def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information
    medical_info = {
        "Age Degeneration": {
            "report": "The patient appears to have age-related degeneration.",
            "preventative_measures": [
                "Further evaluation and management are recommended to prevent vision loss."
                "Regular eye exams are crucial for early detection and intervention",
                "Maintain a healthy lifestyle with a balanced diet and regular exercise",
                "Protect eyes from UV rays with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Schedule regular follow-ups with an eye specialist",
                "Consider supplements recommended by your doctor to support eye health",
            ],
        },
        "Cataract": {
            "report": "It seems like the patient has cataracts.",
            "preventative_measures": [
                "While common and treatable, it's important to address symptoms and consider treatment options."
                "Protect eyes from UV exposure with sunglasses",
                "Quit smoking if applicable, as it can increase cataract risk",
                "Maintain overall health with a balanced diet and regular exercise",
            ],
            "precautionary_measures": [
                "Consult with an eye specialist for personalized treatment options",
                "Discuss surgical options if cataracts significantly affect daily activities",
            ],
        },
        "Diabetes": {
            "report": "The patient appears to have diabetes.",
            "preventative_measures": [
                "It's crucial to manage blood sugar levels effectively to prevent complications."
                "Monitor blood sugar levels regularly as advised by your doctor",
                "Follow a diabetic-friendly diet rich in fruits, vegetables, and whole grains",
                "Engage in regular physical activity to improve insulin sensitivity",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor diabetes management",
                "Consult with an ophthalmologist to assess eye health and discuss preventive measures",
            ],
        },
        "Glaucoma": {
            "report": "The patient shows signs of glaucoma.",
            "preventative_measures": [
                " Early detection and treatment are essential to prevent vision loss."
                "Attend regular eye exams, especially if at risk for glaucoma",
                "Follow treatment plans prescribed by your eye specialist",
                "Manage intraocular pressure through medication or other interventions",
            ],
            "precautionary_measures": [
                "Be vigilant for changes in vision and report them promptly to your doctor",
                "Discuss surgical options if medication alone isn't controlling glaucoma effectively",
            ],
        },
        "Hypertension": {
            "report": "It appears the patient has hypertension.",
            "preventative_measures": [
                "Proper management is crucial to prevent potential eye complications."
                "Monitor blood pressure regularly and follow treatment plans prescribed by your doctor",
                "Adopt a heart-healthy diet low in sodium and high in fruits and vegetables",
                "Engage in regular physical activity to help lower blood pressure",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor blood pressure control",
                "Inform your eye specialist about hypertension diagnosis for comprehensive care",
            ],
        },
        "Myopia": {
            "report": "The patient appears to have myopia.",
            "preventative_measures": [
                "While common, it's important to monitor vision changes and consider corrective measures if needed."
                "Attend regular eye exams to monitor vision changes",
                "Take breaks during prolonged near work to reduce eye strain",
                "Consider corrective lenses or refractive surgery if vision significantly affects daily activities",
            ],
            "precautionary_measures": [
                "Discuss with an eye specialist for personalized recommendations based on severity",
                "Monitor for any progression of myopia and adjust treatment as necessary",
            ],
        },
        "Normal": {
            "report": "Great news! It seems like the patient's eyes are healthy.",
            "preventative_measures": [
                "Regular check-ups are recommended to maintain eye health."
                "Continue with regular eye exams for ongoing monitoring",
                "Maintain overall health with a balanced diet and regular exercise",
                "Protect eyes from UV exposure with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Stay informed about any changes in vision and report them promptly",
                "Schedule annual comprehensive eye check-ups to ensure continued eye health",
            ],
        },
        "Others": {
            "report": "The patient's condition falls into a category not specifically listed.",
            "preventative_measures": [
                "Attend follow-up appointments as advised by your healthcare provider",
                "Discuss any concerns or symptoms with your doctor for appropriate management",
                "Follow recommended lifestyle measures for overall health and well-being",
            ],
            "precautionary_measures": [
                "Seek clarification from your healthcare provider regarding your specific condition",
                "Follow treatment plans or recommendations provided by specialists involved in your care",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = medical_info[predicted_label]["report"]
    preventative_measures = medical_info[predicted_label]["preventative_measures"]
    precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Medical Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )

    precautions = precautionary_measures

    return report, precautions



def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")



def save_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        # Load user data from JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Find the user's information
        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # Convert image bytes to Base64-encoded string
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                # Update the user's information with the Base64-encoded image string
                user_info["Pupil"] = image_base64

                # Save the updated data to JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["Pupil"] = image_base64
                return

        st.error("User not found.")
    except Exception as e:
        st.error(f"Error saving Pupil image to JSON: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "Pupil":None

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    # Custom CSS styles for the form
    form_styles = """
    <style>
    /* Customize text inputs */
    .stTextInput input {
        background-color: #f1f1f1;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus {
        outline: none;
        border-color: #4CAF50;
        box-shadow: 0 0 5px #4CAF50;
    }
    /* Customize password inputs */
    .stTextInput input[type="password"] {
        background-color: #f1f1f1;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stTextInput input[type="password"]:focus {
        outline: none;
        border-color: #4CAF50;
        box-shadow: 0 0 5px #4CAF50;
    }
    /* Customize form submit button */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 10px;
        cursor: pointer;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }
    /* Customize form submit button on hover */
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Customize form title */
    .stTitle {
        color: #4CAF50;
        font-size: 32px;
        margin-bottom: 30px;
    }
    </style>
    """
    st.markdown(form_styles, unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("<div class='stTitle'>Login to Your Account</div>", unsafe_allow_html=True)
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user = check_login(username, password, json_file_path)
            if user is not None:
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None
def generate_pdf_report(user_info):
    # Create a filename based on the current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"medical_report_{timestamp}.pdf"

    # Create a PDF file
    c = canvas.Canvas(file_name, pagesize=letter)

    # Set fonts and font sizes
    title_font_size = 24
    heading_font_size = 16
    text_font_size = 12

    # Draw title
    c.setFont("Helvetica-Bold", title_font_size)
    c.setFillColor(colors.blue)  # Set title color
    c.drawCentredString(300, 750, "Medical Report")

    # Draw user information
    c.setFont("Helvetica-Bold", heading_font_size)
    c.setFillColor(colors.darkgreen)  # Set heading color
    c.drawString(50, 700, "User Information:")
    c.setFont("Helvetica", text_font_size)
    c.setFillColor(colors.black)  # Set text color
    c.drawString(50, 680, f"Name: {user_info['name']}")
    c.drawString(50, 660, f"Age: {user_info['age']}")
    c.drawString(50, 640, f"Sex: {user_info['sex']}")
    today = datetime.date.today().strftime("%d/%m/%Y")
    c.drawString(50, 620, f"Date: {today}")  # Format date as dd/mm/yyyy

    # Draw pupil report
    c.setFont("Helvetica-Bold", heading_font_size)
    c.setFillColor(colors.darkgreen)  # Set heading color
    c.drawString(50, 580, "Pupil Report:")
    c.setFont("Helvetica", text_font_size)
    c.setFillColor(colors.black)  # Set text color
    lines = user_info["report"].split("\n")
    y_coordinate = 560
    for line in lines:
        c.drawString(50, y_coordinate, line)
        y_coordinate -= 20

    # Draw border
    c.setStrokeColorRGB(0, 0, 0)  # Set border color to black
    c.setLineWidth(2)  # Set border width
    c.rect(0, 0, 612, 792, stroke=1, fill=0)

    c.save()
    return file_name 

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        with st.form("Dashboard"):
            st.markdown("<h1 style='color: #4CAF50; font-size: 36px; text-align: center; margin-bottom: 20px;'>Welcome to the Dashboard, {}!</h1>".format(user_info['name']), unsafe_allow_html=True)
            st.markdown("<h2 style='color: #4CAF50; font-size: 24px; margin-bottom: 10px;'>User Information:</h2>", unsafe_allow_html=True)
            st.write(f"Name: {user_info['name']}")
            st.write(f"Sex: {user_info['sex']}")
            st.write(f"Age: {user_info['age']}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Open the JSON file and check for the 'Pupil' key
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                for user in data["users"]:
                    if user["email"] == user_info["email"]:
                        if "Pupil" in user and user["Pupil"] is not None:
                            image_data = base64.b64decode(user["Pupil"])
                            st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Pupil Image", use_column_width=True)
                        if isinstance(user_info["precautions"], list):
                            st.markdown("<h2 style='color: #4CAF50; font-size: 24px; margin-bottom: 10px;'>Precautions:</h2>", unsafe_allow_html=True)
                            for precaution in user_info["precautions"]:
                                st.write(precaution)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='stAlert'>Reminder: Please upload Pupil images and generate a report.</div>", unsafe_allow_html=True)
            # Add a hidden submit button
            st.form_submit_button(label="", help="This button is hidden to prevent 'Missing Submit Button' error")

        # Move the closing div tag outside the form block
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload Pupil images and generate a report."
    )


def main(json_file_path="data.json"):
    uploaded_image = None
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Eye Image", "View Reports"),
        key="PUPILLOMETRY ANALYSIS SYSTEM",
    )
    st.title("PUPILLOMETRY ANALYSIS SYSTEM")
    if page == "Signup/Login":
        st.title("Signup/Login Page")
        css = """
        <style>
        .radio-button-container .radio-button-wrapper {
            display: inline-flex !important;
            flex-direction: row !important;
            justify-content: center !important;
            margin: 0 !important;
        }

        .radio-button-container .radio-button-wrapper .radio-item {
            margin-right: 10px !important;
        }

        .radio-button-container .radio-button-wrapper .radio-item label {
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .radio-button-container .radio-button-wrapper .radio-item label:hover {
            opacity: 0.8;
        }

        .radio-button-container .radio-button-wrapper .radio-item input[type="radio"] {
            display: none;
        }

        .radio-button-container .radio-button-wrapper .radio-item input[type="radio"]:checked + label {
            opacity: 1;
        }

        #Login + label {
            background-color: #5c6bc0;
        }

        #Signup + label {
            background-color: #66bb6a;
        }
        </style>
        """
        # Render CSS
        st.markdown(css, unsafe_allow_html=True)
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)
    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Eye Image":
        if session_state.get("logged_in"):
            st.title("Upload Pupil Image")
            # Model selection
            model_options = ["EfficientNetB0", "VGG16", "VGG19", "DenseNet169", "ResNet50", "Xception", "InceptionV3"]
            model = keras.models.load_model("models\\EfficientNetB0_model.h5")
            selected_model = st.selectbox("Select a model:", model_options)
            if st.button("Load Model"):
                if selected_model == "EfficientNetB0":
                    model = keras.models.load_model("models\\EfficientNetB0_model.h5")
                elif selected_model == "VGG16":
                    model = keras.models.load_model("models\\VGG16_model.h5")
                elif selected_model == "VGG19":
                    model = keras.models.load_model("models\\VGG19_model.h5")
                elif selected_model == "DenseNet169":
                    model = keras.models.load_model("models\\DenseNet169_model.h5")
                elif selected_model == "ResNet50":
                    model = keras.models.load_model("models\\ResNet50_model.h5")
                elif selected_model == "Xception":
                    model = keras.models.load_model("models\\Xception_model.h5")
                elif selected_model == "InceptionV3":
                    model = keras.models.load_model("models\\InceptionV3_model.h5")
                    
                st.success(f"Model {selected_model} has been loaded successfully.")

            # File uploader
            st.title("Upload Pupil Image")
            uploaded_image = st.file_uploader(
                        "Choose a Pupil image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
                    )
            if uploaded_image is not None:
                # Display the image within a centered container
                st.markdown("<div class='centered-image'>", unsafe_allow_html=True)
                st.image(uploaded_image, caption='Uploaded Image')
                st.markdown("</div>", unsafe_allow_html=True)
                if st.button("Predict Condition"):
                    condition = predict(uploaded_image, model)
                    st.write("Predicted Condition: ", condition)
                    st.info("To view the medical report, please navigate to the 'View Reports' section.")

                    # Update user information with the predicted condition
                    user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
                    if user_info is not None:
                        user_info["report"], precautions = generate_medical_report(condition)
                        user_info["precautions"] = precautions
                        with open(json_file_path, "r+") as json_file:
                            data = json.load(json_file)
                            for i, user in enumerate(data["users"]):
                                if user["email"] == user_info["email"]:
                                    data["users"][i] = user_info
                                    break
                            json_file.seek(0)
                            json.dump(data, json_file, indent=4)
                            json_file.truncate()
                    else:
                        st.warning("User information not found.")
        else:
            st.warning("Please login/signup to upload a pupil image.")
    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("User Information:")
                    st.write(f"Name: {user_info['name']}")
                    st.write(f"Age: {user_info['age']}")
                    st.write(f"Sex: {user_info['sex']}")
                    
                    st.subheader("Pupil Report:")
                    st.write(user_info["report"])

                    # Generate PDF report
                    pdf_report_data = generate_pdf_report(user_info)

                    # Add a download button for PDF report
                    download_pdf_button = st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_report_data,
                    file_name="medical_report.pdf",
                    mime="application/pdf"
                    )
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User information not found.")
        else:
            st.warning("Please login/signup to view reports.")


if __name__ == "__main__":
    initialize_database()
    main()
