
# SignMeUp: Real-time recognition of American Sign Language

This project aims to develop a machine learning model that recognizes American Sign Language (ASL) signs in videos. The model is designed to be integrated into an innovative learning app that enables users to practice ASL signs using their phone camera and receive instant feedback. This makes learning ASL more accessible, particularly for the hearing parents of deaf children who may not be familiar with ASL.


# Setup

We started this project from scratch and required several dependencies, which we have outlined in the `requirements.txt` and `requirements_dev.txt` files. To ensure that the project runs correctly, we recommend that you set up a virtual environment before installing the dependencies.

To create a virtual environment, follow these steps:

1. Install `pyenv` to manage your Python versions. You can follow the instructions for your specific operating system on the [pyenv GitHub page](https://github.com/pyenv/pyenv#installation). In the terminal or command prompt, navigate to the project directory and run the command `pyenv local 3.9.8`. This sets the Python version for the current directory to 3.9.8.

2. Create a new virtual environment by running the command `python -m venv .venv`. Activate the virtual environment by running the command `source .venv/bin/activate` on Linux/Mac or `.\\venv\\Scripts\\activate` on Windows.

3. Upgrade `pip` to the latest version by running the command `pip install --upgrade pip`.

4. Install the required dependencies for development by running the command `pip install -r requirements_dev.txt`.


# Installation

To install and run this project, follow these steps:

1. Clone this repository to your local machine.

2. Install the required libraries by running `pip install -r requirements.txt` in your terminal.

3. Download the dataset from [Isolated Sign Recognition Language/Data](https://www.kaggle.com/competitions/asl-signs/data) and save it in the `data` folder.

Note: This project is a Kaggle competition, and therefore requires a Kaggle account to download the dataset. You will need to accept the competition rules and agree to the terms and conditions before you can access the dataset.


# Requirements

The following packages are required to deploy this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- sktime
- TensorFlowLite
- MediaPipe
- OpenCV
- TensorFlow

All these packages can be installed using pip by running the command `pip install -r requirements.txt`.

The following packages are required for development:

- jupyterlab==3.1.14
- matplotlib==3.4.3
- numpy==1.23.1
- pandas==1.3.3
- seaborn==0.11.2
- scikit-learn==1.0
- statsmodels==0.13.2
- pytest==6.2.5
- testbook==0.4.2
- mlflow==1.20.2
- parsenvy==3.0.2
- black==21.9b0
- protobuf==3.20.1

To install these dependencies, run the command `pip install -r requirements_dev.txt`.

# Usage 


Steps to Use ASL Recognition Model:

1. Start the ASL recognition model by running the notebook .... in the designated folder. (I will adjust this part after reorganizing our notebooks into folders). 
2. Ensure that you have good lighting conditions and a clear view of your hand gestures for best results.
3. Point your camera at an ASL sign.
4. Wait for the model to recognize it.
5. The recognized sign will be displayed on the screen.
6. Repeat the process for different ASL signs to practice and improve your skills.

### Note:

- Make sure that the sign is in the dataset used to train the model for accurate recognition.
- If you face any issues with the recognition, try adjusting the lighting or the angle of the camera to get a clearer view of the sign.



# Conclusion

The ASL recognition model developed in this project has the potential to make a significant impact in improving accessibility and communication for the deaf and hard-of-hearing communities. The integration of this model into an innovative learning app has the potential to transform the way people learn ASL, particularly for the hearing parents of deaf children who may not be familiar with ASL.

The next steps for the project involve deploying the model in a mobile application, specifically a sign language learning app. 
Moreover, one of our aim is to collect and add more signs. Additionally, we plan to train models for other sign languages and develop a translation tool that enables natural social interactions and solves everyday problems in public, educational, and commercial settings, such as at a hairdresser or supermarket. The potential impact of these future applications is considerable, making this an exciting and promising project for the advancement of sign language recognition technology.

# Contributing

We welcome contributions from everyone. Here are some ways you can contribute:

- **Report a bug:** If you find a bug in the application, please open an issue on our GitHub repository.
- **Suggest an enhancement:** If you have an idea for a new feature or improvement, please open an issue on our GitHub repository.
- **Contribute code:** If you would like to contribute code to the project, please follow these steps:
  1. Fork the repository
  2. Create a new branch with a descriptive name (`git checkout -b my-new-feature`)
  3. Write your code and tests for your new feature or bug fix
  4. Commit your changes (`git commit -am 'Add some feature'`)
  5. Push to the branch (`git push origin my-new-feature`)
  6. Open a pull request on our GitHub repository

Thank you for your interest in contributing to our project!














