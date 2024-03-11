# Use the official Python image as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY ./app .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using uvicorn (you might need to adjust based on your app structure)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
