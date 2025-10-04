# Feature Extraction

## Dockerized Webcam Demo

> NOTE: The demo has been tested on an LG laptop running Ubuntu 20.04 with CLion 2025.2.3.

Build the release image containing the webcam demo,

    ./building/local/build_image.sh -t release

Run the demo,

    ./building/local/run_webcam_demo.sh

Press any key to close the window and end the demo. 

## Local Development

The project is developed using the CLion IDE. Thanks to
Jetbrains [CLion is now free for non-commercial use](https://blog.jetbrains.com/clion/2025/05/clion-is-now-free-for-non-commercial-use/).
Download it and set it up, it makes C++ development so much easier.

### Toolchain Setup

Use the [Docker toolchain](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html) provided by CLion.
First build the project's development Docker image,

    ./building/local/build_image.sh

At the end you should see `Build successful: feature-extraction:development`.

Then navigate to the CLion toolchain menu ("Settings" > "Build, Execution, Development" > "Toolchains"). Add a toolchain
by hitting the "Add" symbol `+`. In the dropdown menu select "Docker" as the toolchain type. Most of the settings for
the toolchain will self populate. The only two that we need to chang are the "Name" and "Image". The name you select
should be easy to recognize and differentiate it from other toolchains you might add in the future (the toolchains are
set globally across all CLion environments on your computer). Under the image dropdown you should select the development
image we just built: "feature-extraction:development". Then hit the "OK" button, the toolchain setup is complete.

### CMake Setup

Navigate to the CMakeLists found in the `code/` folder, right click on it and select "Load CMake Project" from the menu.
If for some reason this option is not shown, please make sure CLion and all CLion plugins are update, then close the
IDE, delete the `.idea/` folder, reopen CLion and try again. Because our projects do not have the CMakeLists in the root
folder by default, but instead inside of `code/` it might take some fiddling around to work.

Once the CMake project has been loaded you will see that a folder "cmake-build-debug" has been generated. This is not
what we want! This build directory is using the local environment and not the Docker toolchain that we want to use.

To configure our CMake project to use the Docker toolchain, go to the CLion CMake settings menu ("Settings" > "Build,
Execution, Development" > "CMake"). There you will already see the default "Debug" CMake profile. The only setting we
need to chain is the "Toolchain" setting, there from the dropdown menu you should select the toolchain that we created
in the previous step. Then hit the "OK" button, the CMake project setup is complete.

You should now see a second automatically generated folder, something like "
cmake-build-debug-docker-feature-extraction-development". This is the building workspace that CLion will use now, and
you can delete the other auto generated folder that will no longer be used.

To confirm your CMake project setup is complete in the top bar you can select the detected configurations and build,
run, or debug them. Select the "All CTest" configuration and then hit the green play button to "Run" the "All CTest"
configuration. You should see all tests execute and pass in the bottom window of the IDE.

In the "Run / Debug Configurations" dropdown menu where we selected "All CTest" we can also select all other targets (
tests, libraries and executables) in the project's CMakeLists to build them individually. This is extremely practical
when working on large integrated projects.

### The Webcam Example 

> NOTE: The demo has been tested on an LG laptop running Ubuntu 20.04 with CLion 2025.2.3.

To get the webcam example to work we need to change both the toolchain and CMake configuration. 

Go to the toolchain settings ("Settings" > "Build, Execution, Development" > "Toolchains"), make sure the feature 
extraction tool chain we built earlier is selected, and select the gear shaped "Browse" button for the "Container 
Settings" field. In the pop up window copy and paste the below settings in the "Run options" field, ignoring any already 
set options.

    -e DISPLAY=:0.0 --entrypoint= -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged --rm

Hit "OK" to exit the container settings window and "OK" again to apply the settings to the toolchain and exit 
the settings. 

Now with the toolchain setup properly, open the "Run / Debug Configurations" dropdown at the top of the window. At the 
very bottom of the dropdown window select "Edit Configurations..." In the window that opens select the "webcam_demo" 
configuration. In the "Program arguments" field enter "-c target_config.yaml", this is the command line argument which 
is passed to the executable to tell it where the configuration file is located. In the "Working directory" field you 
should select the "Browse..." file explorer icon and select the repositories `code/examples` folder. Then hit the "OK" 
button to save the setting and close the window.

Now we can run the webcam demo! Select the "webcam_demo" configuration in the "Run / Debug Configurations" dropdown and 
then hit the green play button "Run 'webcam_demo'". Hold the target which corresponds to the target configured in 
`code/examples/target_config.yaml` (AprilGrid3 by default) and observe the detections.



