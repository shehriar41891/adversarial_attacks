#FGSM and Gaussian FGSM Adversarial Attacks on ResNet18

This project demonstrates adversarial attacks on the ResNet18 model using the Fast Gradient Sign Method (FGSM) and Gaussian Perturbations to alter the predictions of a deep learning model. It provides a frontend interface using Streamlit and a backend API built with FastAPI.

##Running the Application

 ###Start the Backend API Server
To run the backend server (FastAPI), open a terminal and navigate to the project folder. Then, run:
uvicorn app_fgsm:app --reload


 ###Start the Frontend Streamlit UI
Once the backend is up, run the frontend UI using Streamlit. In another terminal window, run:
streamlit run ui_fgsm.py

##Using the Application
Uploading an Image
Upload an Image: Click the "Upload an Image" button to select a .jpg, .jpeg, or .png image from your local system.

Adjust Epsilon: Use the slider to adjust the perturbation strength (epsilon), which controls the magnitude of the adversarial noise.

Choose Attack Type: You can choose between FGSM and Gaussian FGSM to perform the respective attack.

Enter Ground Truth Label: Provide the ImageNet index (0–999) corresponding to the image's true label.

##Screenshots
###Adversarial Attack using FGSM

![alt text](<./screenshots/Screenshot (602).png>)

###Adversarial Attack using Gaussian Perturbations

![alt text](<./screenshots/Screenshot (604).png>)