# Note

# PI Deployment
For deployment on teh Raspberry PI, the *deployment_personalizer.py* script is not required. Also, the import in the deployment scritps must be modified adequately to run adequately with the actual path on the Pi.

# Pixel Deployment
> - Install Android Debug Bridge on Ubuntu 22.04
> - install termux on the phone from: https://f-droid.org/en/packages/com.termux/  (specifically, press on the Download APK button under the latest specific version, 29th of May for the time of this project)
> - Install this repo: https://github.com/termux-user-repository/tur via

```bash
pkg install tur-repo
pkg update
pkg install python3.10
```

to get the desired version of python: 3.10 for this project. Then every time referring to python/pip, use the suffix python3.10/pip3.10