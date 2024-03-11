# Vision Correction Platform for Smart Glasses

## Overview

This aims to develop a vision correction platform using locally fine-tuned language models (LLM) such as Mistral and LLAVA. The goal is to provide captioning for image inputs from ESP32 cameras, creating a valuable tool for visually impaired individuals. The platform returns text strings, allowing for seamless integration into smart glasses to enhance the visual experience of the blind and contribute to an improved quality of life.


## Features

- **Live Captioning**: The platform captures frames from an ESP32 camera stream in real-time and generates descriptive text captions using locally fine-tuned language models.

- **Local Fine-tuned Language Models (LLM)**: Mistral and LLAVA are employed for their capabilities in understanding and describing visual content, making them suitable for image captioning tasks.

- **ESP32 Camera Integration**: The platform is designed to easily integrate with ESP32 cameras, enabling a cost-effective and accessible solution for vision correction.

- **Smart Glass Compatibility**: The generated text strings can be seamlessly integrated into smart glasses, providing on-the-fly visual descriptions to users.

## Technical Details

### Dependencies

- **ESP32 Libraries**: Utilize the necessary libraries for ESP32 camera interfacing.

- **OpenCV**: Employ OpenCV for image processing tasks, such as capturing frames and preprocessing images before sending them to the language models.

- **LLAVA and Mistral**: Fine-tuned language models for image captioning. Implement these models using appropriate Python libraries (e.g., Hugging Face Transformers)

### Implementation Steps

1. **ESP32 Camera Setup**: Configure the ESP32 camera to stream video frames.

2. **FastAPI Server Setup**: Create a FastAPI server to handle HTTP requests. Implement routes for receiving images, processing them, and returning captions.

3. **Image Processing**: Use OpenCV to capture frames from the ESP32 camera stream. Preprocess the images as needed for better model compatibility.

4. **LLAVA and Mistral Integration**: Incorporate the locally fine-tuned LLMs (LLAVA and Mistral) for image captioning. Ensure that the models are optimized for real-time inference.

5. **WebSocket Integration**: Implement a WebSocket route to provide a live feed for captioning. This allows for continuous updates on the smart glasses.

6. **Smart Glasses Integration**: Develop a smart glasses application that can receive and display the generated text captions.

## Getting Started

