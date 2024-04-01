import pyaudio

p = pyaudio.PyAudio()
file_name = "audio_devices.txt"
audio_devices_file = open(file_name, "w")

print(" > GENERATING...")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    dev_name = dev['name']

    if 'vb' in dev_name.lower():
        line = f"Device {i}: {dev_name}\n"
        audio_devices_file.write(line)
print(f" > FINISHED GENERATING. SEE '{file_name}' FOR MORE DETAILS")