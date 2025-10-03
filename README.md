# Feature Extraction

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
configuration. You should see all tests exectute and pass in the bottom window of the IDE.

-e DISPLAY=:0.0 --entrypoint= -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged --rm

Program arguments: -c target_config.yaml
Working directory: <example directory>



