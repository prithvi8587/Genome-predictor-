# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the cloud container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all application files (including your templates folder)
COPY . .

# Expose the network port
EXPOSE 10000

# Command to run the application
CMD ["python", "app.py"]

