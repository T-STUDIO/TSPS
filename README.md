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
