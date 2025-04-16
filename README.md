
# Start Flask API
python app.py

# Start Ghrok Server
ngrok http 5000

# Creating new virtual nev
1. python3.10 -m venv venv
2.(MAC) source venv/bin/activate | (Windows) venv\Scripts\Activate



deactivate


# Dependencies
pip install git+https://github.com/facebookresearch/audiocraft.git




# requirements.txt
## Update requirements.txt
pip freeze > requirements.txt

## Install requirements.txt
pip install -r requirements.txt




hypercorn api.app:app --bind 0.0.0.0:5000

