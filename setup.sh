mkdir -p ~/.streamlit/
echo "[general]
email = \"gupta123vaibhav@live.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml