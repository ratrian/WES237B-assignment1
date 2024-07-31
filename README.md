# Assignment 1

## Setup

### F-Droid
Using Google Chrome on your Pixel Fold, navigate to the F-Droid [website](https://f-droid.org/en/).  Click on the `DOWNLOAD F-DROID` button.  This will download the `apk` for F-Droid.  Look for the notification for the download.  Once you tap it, you will be prompted with the screen to enable installation of `apk`s from Google Chrome.  Turn on the setting and continue.

### Termux
Open F-Droid and search for `Termux`.  Install the package `Termux: Terminal Emulator with Packages`.  this will once again prompt you to enable the installation of `apk`s from F-Droid.  Turn on the setting and continue.  You will then be warned that `Termux` is intended for an older version of Android.  Click `More Information` and select `Install anyway`.  Start `Termux`, accept notifications.  Close `Termux`.  Start `Termux` again.  Check your notifications, expand the `Termux` notification and click `Aquire wake lock`.

### Install Code Server
In `Termux` perform the following
1. `pkg update`
2. `pkg upgrade`.  Accept all of the defaults
3. `pkg install tur-repo`
4. `pkg install code-server`

### Launch Code Server
Run the following command to start the `code-server`:

* With authentication: `code-server --bind-addr 0.0.0.0:8080, find the password in /code-server/config.yaml`
* Without authentication: `code-server --auth none --bind-addr 0.0.0.0:8080`

Navivate to Settings -> About phone.  Look for your IPv4 address.

In your browser on your laptop, navigate to `http:<IPv4>:8080`.

### Downloading the Repo
In `code-server` in your browser, open up terminal by selecting the Hamburger menu->View->Terminal.  Run the following commands
1. `pkg install git`
2. `ssh-keygen`.  Accept the defaults.
3. `cat ~/.ssh/id_ed25519.pub`.  Add this ssh key to your GitHub account.
4. `git clone git@github.com:ratrian/WES237B-assignment1.git`.  Open the folder in `code-server` using the Hamburger menu -> File -> Open Folder.
5. `pkg install build-essential`
6. `pkg install binutils`
7. `pkg install ocl-icd`
8. `pkg install opencl-headers`
9. Add `export LD_LIBRARY_PATH=/system/vendor/lib64/egl:/system/vendor/lib64:$LD_LIBRARY_PATH` to your `~/.bashrc`
10. `source ~/.bashrc`.  You will only need to do this until you restart your phone.
11. `git submodule update --init --recursive`
12. `cd ./lab/device_query`
13. `make`
14. `./device_query`

## Lab

### Device Query
The purpose of this portion of the lab is to familiarize the student with the underlying hardware and how it relates to the OpenCL hardware model.  By extension, the program will familiarize the student with the processing resources at hand along with their capabilities.  By the end of this lab, one should understand the process of compiling and running code that will be used in subsequent modules.

The instructions for this portion of the lab is located in `lab/device_query/README.md`.

### Raytracer Sequential 
This is a ray tracer program that creates `output.png` using your CPU sequentially.  Examine `main.c` and understand what it does.  Run this program with the `time` command and report the results as part of your report for the assignment.

### Raytracer Parallel
This is a ray tracer program that creates `output.png` using your CPU as an OpenCL device. Examine `main.c` and `kernel.cl`. Run this program with the `time` command and report the results as part of your report for the assignment.

Think about why the results from `time` differ.  What can this tell you about OpenCL?  Report your thoughts in your report.

## Homework
The purpose of this lab is to introduce the student to the OpenCL API by implementing vector addition. The student will implement vector addition by writing the OpenCL kernel code as well as the associated host code.

The instructions for this portion of the homework is located in `homework/vector_add/README.md`.

## Resources
* [Documentation](https://www.khronos.org/opencl/)