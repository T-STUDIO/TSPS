# Local Astrometry.net & ASTAP API Web App

Requires setting up Astrometry.net, sep, and ASTAP in a local environment.

# Astrometry.net
sudo apt install -y astrometry.net

sudo apt install -y sextractor || sudo apt install -y source-extractor
sudo apt install pipx

# ASTAP
https://www.hnsky.org/astap.htm


# TSPS
Local Astrometry.net API web app

Requires a local Astrometry.net.

sudo apt install pipx

pipx ensurepath

pipx install uvicorn

pipx inject uvicorn fastapi python-multipart Pillow numpy astropy
   
https://tstudioastronomy.blog.fc2.com/blog-category-46.html

（Japanese Only)

# Automatic Start

sudo nano /etc/systemd/system/solver_server.service

[Unit]
Description=TAWS Solver Server (pipx environment)
After=network.target

[Service]
User=astrpi64
WorkingDirectory=/your-dir/ps/

Environment="PATH=/usr/local/bin:/usr/bin:/bin"

ExecStart=/home/astrpi64/.local/bin/uvicorn solver_server:app --host 0.0.0.0 --port 6001

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

→Save

sudo systemctl enable solver_server.service
sudo systemctl start solver_server.service
