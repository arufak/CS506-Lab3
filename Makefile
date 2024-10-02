# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Install dependencies
install:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	set FLASK_APP=app.py && set FLASK_ENV=development && venv\Scripts\flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install