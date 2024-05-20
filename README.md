# Eye Health Station

Eye Health Station is a web application designed to analyze pupil images using machine learning techniques to detect various eye conditions. It provides personalized medical reports and recommendations based on uploaded images, with user authentication for a tailored experience.

## Features

- **User Authentication:**
  - Signup/Login: Create an account or log in to access the dashboard.
  
- **Dashboard:**
  - Displays user details, uploaded images with reports, and personalized recommendations.
  
- **Upload Eye Image:**
  - Supports multiple pre-trained models for image analysis, predicting eye conditions, and generating detailed reports.

- **View Reports:**
  - Enables users to access and review detailed medical reports from previous uploads.

## Technical Stack

- **Frontend:** Streamlit
- **Backend:** TensorFlow, Keras
- **Storage:** JSON files
- **Image Processing:** PIL

## Workflow

1. **Authentication:** Users sign up or log in to access their account.
2. **Image Upload:** Users upload pupil images, which are processed by the selected model.
3. **Prediction and Report:** The system predicts the eye condition and generates a medical report with recommendations.
4. **Dashboard:** Users view their information, uploaded images, and reports.

## Use Cases

- **Healthcare Providers:** For preliminary patient screening.
- **Individuals:** To monitor eye health and receive actionable insights.

## Future Enhancements

- **Database Integration:** Transition to a more robust database system.
- **Expanded Conditions:** Include models for a wider range of eye conditions.
- **Mobile Compatibility:** Develop a mobile app for easier access.
