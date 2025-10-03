# Feature Extraction

## Local Development

The project is developed using the CLion IDE. Thanks to
Jetbrains [CLion is now free for non-commercial use](https://blog.jetbrains.com/clion/2025/05/clion-is-now-free-for-non-commercial-use/).
Download it and set it up, it makes C++ development so much easier.

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



-e DISPLAY=:0.0 --entrypoint= -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged --rm

Program arguments: -c target_config.yaml
Working directory: <example directory>



