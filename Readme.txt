# Step-by-Step Guide

**IMPORTANT**: BEFORE ATTEMPTING ANY TESTS, READ THIS FILE IN ITS ENTIRELY. APPROACH THE TESTS ONLY AFTER YOU HAVE FULLY READ AND UNDERSTOOD THIS DOCUMENT AT LEAST ONCE.

## Preliminary Steps

1. **Download and Install the VB-Audio Virtual Cable Driver**

    This software allows the use of a virtual audio cable, enabling any application to be hooked up to the input end of the cable, thus permitting us to read from the output end.
    
    - We will not provide detailed installation instructions here, as it follows the standard procedure of installing software.
    
    - **NOTICE**: If you own a physical audio cable, the testing can still proceed similarly.

2. **Run `get_VB_Audio_devices.py` File**

    After successfully executing this program, a text file will be generated, listing all devices on the current machine that contain "VB-Audio" in their names, alongside their respective indexes.

    - If the text file is empty, the driver from step 1 likely wasn't installed correctly.

    - **IMPORTANT**: The indexes may not remain accurate over time (even if the computer hasn't been turned off since it was last run). It's recommended to always rerun this script before performing any tests to ensure the device index is correct.

## Testing Steps

3. **Select the Correct OUTPUT Device Number**

    From the `audio_devices.txt` file generated, identify the correct device number through trial and error—preferably starting with the lowest index as it's often the correct device.
    
    For example:
    ```
    Device 3: CABLE Output (VB-Audio Virtual Device)
    Device 5: CABLE Input (VB-Audio Virtual Cable)
    Device 10: CABLE Output (VB-Audio Virtual Cable)
    ```
    - Start testing with Device 3, then move to Device 10 if the former doesn't work (skip Device 5 since it's an input-end, whereas an output-end is required).

4. **Configure the Correct Device in the Test Script**

    Open the script you wish to test, find the line initializing the device index, and replace it with the index verified in the previous step.
    
    For instance:
    ```python
    device_index = 5
    ```
    Change to:
    ```python
    device_index = 3 # Or whichever number you've found to be correct
    ```
    - This part requires trial and error initially, so be prepared to adjust and retry as necessary.

5. **Set Computer's Audio Output**

    Ensure the computer's audio output is set to the input-end of the VB-Audio Virtual Cable before running the tests.

6. **Execute the Model and Model-main Files**

    For testing, say, `RandomForest-main.py`, run `RandomForest.py` **BEFORE** proceeding with the `RandomForest-main.py`.

    - A semi-transparent window with arrows and buttons should appear indicating the test interface.

7. **Start the Test**

    Click on "start" in the window that appeared to initiate the testing procedure.

8. **Monitor for Errors**

    Pay close attention to the console for potential errors:
    
    - Common Error Example:
        ```
        OSError: [Errno -9998] Invalid number of channels
        ```
      If you encounter this, the device index is incorrect. Return to step 3 to select a different index.
