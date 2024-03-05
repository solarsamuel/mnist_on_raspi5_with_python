import subprocess

# Define the command to take a picture
#command = "libcamera-jpeg -o test1.jpg"
#command = "libcamera-jpeg -o test4.png -t 2000 --width 32 --height 32" #DOES NOT work
command = "libcamera-jpeg -o test8.png --width 56 --height 56" #works


# Run the command using subprocess
subprocess.run(command, shell=True)