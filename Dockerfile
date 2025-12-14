# Use Python 3.9
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the code
COPY . .

# Grant permissions (important for Hugging Face)
RUN chmod -R 777 /code

# Run the application on port 7860 (Hugging Face standard)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "server:app"]