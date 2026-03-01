# TSPS
Local Astrometry.net API web app

Requires a local Astrometry.net.

sudo apt install pipx

pipx ensurepath

pipx install uvicorn

pipx inject uvicorn fastapi python-multipart Pillow numpy astropy
