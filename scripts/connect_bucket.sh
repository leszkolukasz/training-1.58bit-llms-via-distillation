mkdir -p ~/data

gcsfuse --implicit-dirs \
    --rename-dir-limit=100 \
    --max-conns-per-host=100 \
    nlp-project-data ~/data